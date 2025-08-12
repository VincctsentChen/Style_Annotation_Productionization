#!/usr/bin/env python3
import os
import re
import json
import traceback
import argparse
import yaml
from pathlib import Path
import pandas as pd
import requests
import random
import logging
from pydantic import BaseModel
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import vertexai
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from requests.exceptions import Timeout as RequestsTimeout, ReadTimeout, ConnectTimeout, RequestException
from gql.transport.exceptions import TransportServerError


# ─── Models & Clients ──────────────────────────────────────────────────────────
class Ocid(BaseModel):
    ocid: int
    name: str
    url: str

class ProductTextInfo(BaseModel):
    prsku: str
    ireid: int
    base_image_url: str
    name: str
    clid: int
    clname: str
    romance_copy: str
    ocids: list[Ocid]
    number_of_ratings: int
    review_rating: float
    is_sale: bool

class ProductCacheClient:
    def __init__(self, base_url, auth_path, graphql_path):
        self.base_url = base_url
        self.auth_endpoint = auth_path
        self.graphql_endpoint = graphql_path
        self.token = None
        self.client = None

    def _refresh_token(self):
        resp = requests.post(f"{self.base_url}{self.auth_endpoint}")
        resp.raise_for_status()
        self.token = resp.json().get("access_token")
        transport = RequestsHTTPTransport(
            url=f"{self.base_url}{self.graphql_endpoint}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.client = Client(transport=transport, fetch_schema_from_transport=True)

    def _ensure(self):
        if not self.client:
            self._refresh_token()

    def get_product_data(self, skus: list[str], brand_catalog_id: int) -> list[ProductTextInfo]:
        self._ensure()
        query = gql(
            '''query ProductQuery($brandCatalog: String!, $skus: [String!]!) {
                products(brandCatalog: $brandCatalog, skus: $skus) {
                  name displaySku isFindable romanceCopy { value }
                  imageResourceId productOptions { id name imageResourceId }
                  productClasses { id name }
                  numberOfRatings reviewRating promotion { flag }
                }
            }'''
        )
        vars = {"brandCatalog": str(brand_catalog_id), "skus": skus}
        
        '''
        The following try/except section deals with timeout situation
        '''
        try:
            resp = self.client.execute(query, variable_values=vars)
        except (RequestsTimeout, ReadTimeout, ConnectTimeout) as e:
            logging.warning(f"[TIMEOUT] get_product_data timed out for skus={skus}: {e}")
            return []  # <-- signal caller to skip this row
        except TransportServerError as e:
            # Many proxies return 504/408 for timeouts; treat them as skip
            code = getattr(e, "code", None)
            if code in (504, 408):
                logging.warning(f"[TIMEOUT {code}] get_product_data for skus={skus}: {e}")
                return []  # <-- skip this row
            raise  # non-timeout server errors should still bubble up
        except RequestException as e:
            # Network-level issues that include timeouts; you can choose to skip as well
            logging.warning(f"[NETWORK] get_product_data request issue for skus={skus}: {e}")
            return []  # <-- optional: skip on any request error
        
        resp = self.client.execute(query, variable_values=vars)
        products = resp.get("products", [])
        results = []         
        for sku, prod in zip(skus, products):
            if prod and prod.get("isFindable"):
                opts = [Ocid(ocid=o["id"], name=o["name"], url=self._image_url(o["imageResourceId"]))
                        for o in prod.get("productOptions", [])]
                info = ProductTextInfo(
                    prsku=sku,
                    ireid=prod.get("imageResourceId"),
                    base_image_url=self._image_url(prod.get("imageResourceId")),
                    name=prod.get("name"),
                    clid=prod["productClasses"][0]["id"],
                    clname=prod["productClasses"][0]["name"],
                    romance_copy=(prod.get("romanceCopy") or {}).get("value", ""),
                    ocids=opts,
                    number_of_ratings=prod.get("numberOfRatings", 0),
                    review_rating=prod.get("reviewRating", 0.0),
                    is_sale=bool(prod.get("promotion", {}).get("flag", False)),
                )
                results.append(info)
        return results

    @staticmethod
    def _image_url(img_id: int) -> str:
        return f"https://secure.img1-fg.wfcdn.com/lf/unprocessed/hash/0/{img_id}/1/c.jpg"

class GeminiClient:
    def __init__(self, model_name: str):
        vertexai.init(project=os.environ.get("GCP_PROJECT"), location="us-central1")
        self.model = GenerativeModel(model_name)
        self.gen_config = {"max_output_tokens": 4096, "temperature": 0.1}
        self.safety = [
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                  threshold=SafetySetting.HarmBlockThreshold.OFF),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                  threshold=SafetySetting.HarmBlockThreshold.OFF),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                  threshold=SafetySetting.HarmBlockThreshold.OFF),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                  threshold=SafetySetting.HarmBlockThreshold.OFF),
]

    def image_and_text(self, urls: list[str], system_prompt: str, task_prompt: str) -> str:
        parts = [Part.from_uri(uri=u, mime_type=('image/png' if u.lower().endswith('.png') else 'image/jpeg')) for u in urls]
        content = 'System: ' + system_prompt.strip() + '\n\n' + 'Task: ' + task_prompt.strip()
        return self.model.generate_content([content] + parts, generation_config=self.gen_config, safety_settings=self.safety).text.strip()

# ─── Annotation Logic ──────────────────────────────────────────────────────────
def annotate(sku1, sku2, examples_csv_path: str, gemini: GeminiClient) -> tuple[dict,str,str]:
    # 1) load & format few-shot examples exactly as before
    df = pd.read_csv(examples_csv_path)
    examples_blocks = []
    for idx, row in df.iterrows():
        label = "Compatible" if row["compatible"].strip().lower() == "yes" else "Not Compatible"
        block = (
            f"EXAMPLE {idx+1} ({label}):\n"
            f"  • Image A: {row['image_a_url']}\n"
            f"    Product A: {row['product_a']}\n"
            f"  • Image B: {row['image_b_url']}\n"
            f"    Product B: {row['product_b']}\n"
            f"  → Compatible: {row['compatible']}\n"
            f"  Reason: {row['reason']}\n"
        )
        examples_blocks.append(block)
    examples_section = "\n\n".join(examples_blocks)

    # 2) system prompt with your full Definition of compatibility
    system_prompt = f"""
    You are a professional interior designer and style advisor for Wayfair, specialized in room design.
    Your job is to decide if two products “go together” stylistically.

    **Definition of compatibility**
    - **Yes**:
        1. Two products can be placed in the same room without creating visual tension or clashing aesthetics. They share a common design language in terms of shape, materials, colors, or lineage—such that they feel part of a cohesive, intentional interior design.
        2. **Items of the *same category* (e.g. two dining tables)** are compatible if they share key style attributes (finish, silhouette, scale) and would look harmonious side by side, regardless of minor functional differences (height, capacity).
    - **No**:
        1. Two products display clearly conflicting visual languages such that placing them together would create a jarring or incoherent aesthetic. They do not share a unifying theme.
        2. They, despite matching in style, serve fundamentally different functions that cannot coexist in the same room (e.g., bathroom vanity + coffee table).

    Below are EXAMPLES of what a compatible vs. non-compatible pairing looks like, with images provided:

        {examples_section}

        When you see PRODUCT 1 & PRODUCT 2 below:
        1. List 2–3 shared design features (shape, material, color).
        2. Decide if they are Stylistically Compatible (“Yes” or “No”).

        Output **only** valid JSON, for example:
        ```json
        {{
          "compatible": "Yes",
          "reason": "Both pieces share white marble surfaces and clean, classic lines ideal for a modern-transitional bathroom."
        }}
        ```"""

    # 3) task prompt exactly as in the notebook
    def detail_prompt(sku):
        ocid_name = f"Option Detail: {sku.ocids[0].name}" if sku.ocids else ""
        url        = sku.ocids[0].url if sku.ocids else sku.base_image_url
        meta = (
            f"Title: {sku.name}\n"
            f"Class Name: {sku.clname}\n"
            f"Description: {sku.romance_copy}\n"
            f"{ocid_name}".strip()
        )
        return url, meta

    img1, meta1 = detail_prompt(sku1)
    img2, meta2 = detail_prompt(sku2)

    task_prompt = f"""
    Product 1:
    {meta1}

    Product 2:
    {meta2}
    """
    # 4) call Gemini and parse JSON
    raw = gemini.image_and_text([img1, img2], system_prompt, task_prompt)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    payload = m.group(1) if m else raw

    # DEBUG: print out what we got back
    print("=== RAW RESPONSE ===")
    print(raw)
    print("=== JSON PAYLOAD ===")
    print(repr(payload))

    try:
        annotation = json.loads(payload)
    except json.JSONDecodeError as e:
        print("❌ JSON decode failed:", e)
        raise

    return annotation, img1, img2



def read_input_table(file_path: str, sheet: str | int | None = None) -> pd.DataFrame:
    '''
    Read .csv or .xlsx files.
    '''
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in {".xlsx", ".xls"}:
        if ext == ".xlsx" and not zipfile.is_zipfile(file_path):
            raise ValueError(f"{file_path} has .xlsx extension but is not a valid Excel file (zip).")
        try:
            return pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
        except ValueError:
            # fallback to first sheet if the requested sheet name is missing
            xls = pd.ExcelFile(file_path, engine="openpyxl")
            return pd.read_excel(file_path, sheet_name=xls.sheet_names[0], engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def process_labeled(file_path, sheet, examples_csv, cache, gemini, brand_id):
    '''
    This function compares the human annotator's label and Gemini label
    '''
    print("Start the comparision mode")
    df = read_input_table(file_path, sheet)
    #df = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
    records = []
    for idx, row in df.iterrows():
        sku1, sku2 = row["product_1"], row["product_2"]
        data1 = cache.get_product_data([sku1], brand_id)
        data2 = cache.get_product_data([sku2], brand_id)
        if not data1:
            logging.error(f"SKU not found in cache: {sku1} (row {idx})")
            continue
        if not data2:
            logging.error(f"SKU not found in cache: {sku2} (row {idx})")
            continue
        p1 = data1[0]
        p2 = data2[0]

        ann, img1, img2 = annotate(p1, p2, examples_csv, gemini)
        records.append({
            "product_1": sku1,
            "product_1_title": p1.name,
            "product_1_img": p1.base_image_url,   # fallback just in case
            "product_2": sku2,
            "product_2_title": p2.name,
            "product_2_img": p2.base_image_url,   # fallback just in case
            "human": row["Compatible?"],
            "ai": ann.get("compatible"),
            "ai_reason": ann.get("reason", "")
        })
    
    out = pd.DataFrame(records)
    cm = pd.crosstab(out["human"].str.lower(), out["ai"].str.lower(),
                     rownames=["human"], colnames=["ai"])
    print("\nConfusion Matrix:\n", cm)
    
    cm_dir = cfg["outputs"]['confusion_matrix_dir'] # confusion_matrix_dir
    stem = Path(file_path).stem  # e.g., "bathroom_1"
    m = re.search(r"([A-Za-z]+)_\d+$", stem)   # e.g., "bathroom"
    room = m.group(1) if m else stem  # fallback to full stem if pattern doesn't match
    if cm_dir:
        os.makedirs(cm_dir, exist_ok=True)
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_title(f"Confusion Matrix: {room}")
        fig.tight_layout()
        out_png = os.path.join(cm_dir, f"{room}_cm.png")
        fig.savefig(out_png, dpi=150)
        print(f"[INFO] Saved confusion matrix → {out_png}")
        plt.close(fig)
    
    return out

def evaluate_unlabeled(file_path, sheet, examples_csv, cache, gemini, brand_id):
    '''
    This function generates labels for new data
    '''
    print("Start the evaluation mode")
    df = read_input_table(file_path, sheet)
    #df = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")

    # initialize empty columns so lengths always match even if we skip rows
    df["pred_compatible"] = pd.NA
    df["pred_yes"] = pd.NA
    df["pred_reason"] = pd.NA

    for idx, row in df.iterrows():
        sku1, sku2 = row["product_1"], row["product_2"]
        data1 = cache.get_product_data([sku1], brand_id)
        data2 = cache.get_product_data([sku2], brand_id)
        if not data1:
            logging.error(f"SKU not found in cache: {sku1} (row {idx})")
            continue
        if not data2:
            logging.error(f"SKU not found in cache: {sku2} (row {idx})")
            continue
        p1, p2 = data1[0], data2[0]

        ann, _, _ = annotate(p1, p2, examples_csv, gemini)
        comp = (ann.get("compatible") or "").strip()
        reason = ann.get("reason", "")

        df.loc[idx, "pred_compatible"] = comp
        df.loc[idx, "pred_yes"] = (comp.lower() == "yes")
        df.loc[idx, "pred_reason"] = reason

    # compute % yes over rows we actually predicted
    mask = df["pred_yes"].notna()
    pct = float(df.loc[mask, "pred_yes"].mean() * 100) if mask.any() else 0.0
    print(f"\nEvaluated {int(mask.sum())} pairs; % yes = {pct:.1f}%")

    return df


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", required=True)
    parser.add_argument("-m","--mode",
                   choices=["compare","evaluate"],
                   help="Override mode from config (run.mode)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = args.mode or (cfg.get("run") or {}).get("mode", "compare")
    print(f"[INFO] Running mode: {mode}")

    # Initialize clients
    bc = ProductCacheClient(cfg['service']['base_url'], cfg['service']['auth'], cfg['service']['graphql'])
    gemini = GeminiClient(cfg['gemini']['model_name'])

    data_dir = cfg['inputs']['data_dir']
    examples_csv_path = cfg['inputs']['examples_csv_path']
    output_dir = cfg['outputs']['results_dir']
    rooms = cfg.get('rooms', [])
    brand_id = cfg['service'].get('brand_catalog_id', 1)
    
    # NEW: where to save confusion matrices
    cm_dir = os.path.join(output_dir, "Confusion_Matrix")

    os.makedirs(output_dir, exist_ok=True)
    # Iterate over .xlsx and .csv in the input folder
    for infile_path in Path(data_dir).glob("*"):
        if infile_path.suffix.lower() not in {".xlsx", ".csv"}:
            continue  # skip non-data files

        infile = str(infile_path)
        stem = infile_path.stem

        # Default: no sheet for CSV
        sheet = None

        if infile_path.suffix.lower() == ".xlsx":
            try:
                xls = pd.ExcelFile(infile, engine="openpyxl")
                sheet = stem if stem in xls.sheet_names else xls.sheet_names[0]
            except Exception as e:
                print(f"[WARN] Could not read Excel sheets for {infile}: {e}")
                sheet = None

        # Now run in chosen mode
        if mode == "compare":
            df_out = process_labeled(infile, sheet, examples_csv_path, bc, gemini, brand_id)
            out_csv = os.path.join(output_dir, f"{stem}_compare.csv")
            df_out.to_csv(out_csv, index=False)
            print(f"[OK] Wrote {len(df_out)} rows → {out_csv}")
        else:
            df_eval = evaluate_unlabeled(infile, sheet, examples_csv_path, bc, gemini, brand_id)
            out_csv = os.path.join(output_dir, f"{stem}_eval.csv")
            df_eval.to_csv(out_csv, index=False)
            print(f"[OK] Wrote {len(df_eval)} rows (with predictions) → {out_csv}")
    
    
#     # Iterate over every .xlsx in the input folder
#     for infile_path in Path(data_dir).glob("*.xlsx"):
#         infile = str(infile_path)
#         stem   = infile_path.stem

#         # Pick sheet: try a sheet that matches the file name; fallback to the first sheet
#         try:
#             xls = pd.ExcelFile(infile, engine="openpyxl")
#             sheet = stem if stem in xls.sheet_names else xls.sheet_names[0]
#         except Exception:
#             # fallback if listing sheets fails
#             sheet = stem

#         if mode == "compare":
#             df_out  = process_labeled(infile, sheet, examples_csv_path, bc, gemini, brand_id)
#             out_csv = os.path.join(output_dir, f"{stem}_compare.csv")
#             df_out.to_csv(out_csv, index=False)
#             print(f"[OK] Wrote {len(df_out)} rows → {out_csv}")
#         else:
#             df_eval = evaluate_unlabeled(infile, sheet, examples_csv_path, bc, gemini, brand_id)
#             out_csv = os.path.join(output_dir, f"{stem}_eval.csv")
#             df_eval.to_csv(out_csv, index=False)
#             print(f"[OK] Wrote {len(df_eval)} rows (with predictions) → {out_csv}")


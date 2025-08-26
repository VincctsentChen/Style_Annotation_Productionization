# Style Annotation Productionization

This repository contains code for an **LLM-powered furniture style annotation pipeline**, 
developed during my internship as a Machine Learning Scientist at Wayfair.  

The system applies **Gemini** (Google Cloud Vertex AI) to automatically annotate furniture 
styles and determine whether two pieces are **stylistically compatible** 
(i.e., whether they “fit together” in the same room). Applying LLMs for labeling offers an advantage over traditional ML by eliminating the need for manual feature engineering.   
Once finalized, the LLM agent can automatically label style compatibility, greatly **reducing reliance on human annotators** and **saving both time and cost**.

This work is detailed in my Wayfair Tech Blog article:  
👉 [Teaching Wayfair’s Catalog to See Style: An LLM-Powered Style Compatibility Labeling Pipeline on Google Cloud](https://www.aboutwayfair.com/careers/tech-blog/teaching-wayfairs-catalog-to-see-style-an-llm-powered-style-compatibility-labeling-pipeline-on-google-cloud)

---

## Getting Started
The master script is 'StyleAnnotation_four_rooms_Gemeni.py'.  
1. **Clone** the repo and enter the project directory.
2. **Run the master script**:
   ```bash
   python StyleAnnotation_four_rooms_Gemeni.py --config config.yaml


## Modes
- **compare**: compare LLM annotation vs. human annotation
- **evaluate**: run evaluation on an unlabeled dataset and report % 'Yes'

You can change the mode argument in the `config.yaml` file.

## Repository Structure  
- StyleAnnotation_four_rooms_Gemeni.py — main entry point for the pipeline.  
- config.yaml — configuration file for mode, dataset paths, and model parameters.  
- data/ — contains input product metadata and style-pair datasets (not included).  
- outputs/ — stores generated annotations and evaluation results.  

## Tech Stack  
- LLM: Gemini on Google Cloud Vertex AI  
- Storage & Processing: Google Cloud Storage + BigQuery  
- Evaluation: Human-labeled expert dataset for benchmarking


 


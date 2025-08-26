# Style_Annotation_Productionization

The script applies LLM (Gemini) to automate furniture style annotation.  
Given a furniture piece, the script can judge whether another furniture piece is stylistically compatible with it.  
The master script is 'StyleAnnotation_four_rooms_Gemeni.py'. 
## Getting Started
1. **Clone** the repo and enter the project directory.
3. **Run the master script**:
   ```bash
   python StyleAnnotation_four_rooms_Gemeni.py --config config.yaml


## Modes
- **compare**: compare LLM annotation vs. human annotation
- **evaluate**: run evaluation on an unlabeled dataset and report % 'Yes'

You can change the mode argument in the `config.yaml` file.


 


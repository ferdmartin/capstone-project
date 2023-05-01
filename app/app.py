def main():
    """
    Creates a Streamlit web app that classifies a given body of text as either human-made or AI-generated,
    using a pre-trained model. 
    """
    import streamlit as st
    import numpy as np
    import joblib
    import string
    import scipy
    import spacy
    import re
    from transformers import AutoTokenizer
    import torch
    from eli5.lime import TextExplainer
    from eli5.lime.samplers import MaskingTextSampler
    import eli5
    import shap
    from custom_models import HF_DistilBertBasedModelAppDocs, HF_BertBasedModelAppDocs

    # Initialize Spacy
    nlp = spacy.load("en_core_web_sm")
    
    # device to run DL model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def format_text(text: str) -> str:
        """
        This function takes a string as input and returns a formatted version of the string. 
        The function replaces specific substrings in the input string with empty strings, 
        converts the string to lowercase, removes any leading or trailing whitespace, 
        and removes any punctuation from the string. 
        """

        text = nlp(text)
        text = " ".join([token.text for token in text if token.ent_type_ not in ["PERSON", "DATE"]])

        pattern = r"\b[A-Za-z]+\d+\b"
        text = re.sub(pattern, "", text)
        
        return text.replace("REDACTED", "").lower().replace("[Name]", "").replace("[your name]", "").\
                                replace("dear admissions committee,", "").replace("sincerely,","").\
                                replace("[university's name]","fordham").replace("dear sir/madam,","").\
                                replace("– statement of intent  ","").\
                                replace('program: master of science in data analytics  name of applicant:    ',"").\
                                replace("data analytics", "data science").replace("| \u200b","").\
                                replace("m.s. in data science at lincoln center  ","").\
                                translate(str.maketrans('', '', string.punctuation)).strip().lstrip()

    # Define the function to classify text
    def nb_lr(model, text):
        # Clean and format the input text
        text = format_text(text)
        # Predict using either LR or NB and get prediction probability
        prediction = model.predict([text]).item()
        predict_proba = round(model.predict_proba([text]).squeeze()[prediction].item(),4)
        return prediction, predict_proba
    
    def torch_pred(tokenizer, model, text):
        # DL models (BERT/DistilBERT based models)
        cleaned_text_tokens = tokenizer([text], padding='max_length', max_length=512, truncation=True)
        with torch.inference_mode():
            input_ids, att = cleaned_text_tokens["input_ids"], cleaned_text_tokens["attention_mask"]
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(att).to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(logits, 1)
            prediction = prediction.item()
            predict_proba = round(torch.softmax(logits, 1).cpu().squeeze().tolist()[prediction],4)
            return prediction, predict_proba

    def pred_str(prediction):
    # Map the predicted class to string output
        if prediction == 0:
            return "Human-made 🤷‍♂️🤷‍♀️"
        else:
            return "Generated with AI 🦾"
    
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def load_tokenizer(option):
        if option == "BERT-based model":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding='max_length', max_length=512, truncation=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", padding='max_length', max_length=512, truncation=True)
        return tokenizer

    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def load_model(option):
        if option == "BERT-based model":
            model = HF_BertBasedModelAppDocs.from_pretrained("ferdmartin/HF_BertBasedModelAppDocs").to(device)
        else:
            model = HF_DistilBertBasedModelAppDocs.from_pretrained("ferdmartin/HF_DistilBertBasedModelAppDocs").to(device)
        return model
    

    # Streamlit app:

    models_available = {"Logistic Regression":"models/baseline_model_lr2.joblib", 
                        "Naive Bayes": "models/baseline_model_nb2.joblib",
                        "DistilBERT-based model (BERT light)": "ferdmartin/HF_DistilBertBasedModelAppDocs",
                        "BERT-based model": "ferdmartin/HF_BertBasedModelAppDocs"
                        }

    st.set_page_config(page_title="AI/Human GradAppDocs", page_icon="🤖", layout="wide")
    st.title("Academic Application Document Classifier")
    st.header("Is it human-made 📝 or Generated with AI 🤖 ?  ")
    
    # Check the model to use
    def restore_prediction_state():
        if "prediction" in st.session_state:
            del st.session_state.prediction
    option = st.selectbox("Select a model to use:", models_available, on_change=restore_prediction_state)
    

    # Load the selected trained model
    if option in ("BERT-based model", "DistilBERT-based model (BERT light)"):
        tokenizer = load_tokenizer(option)
        model = load_model(option)
    else:
        model = joblib.load(models_available[option])


    text = st.text_area("Enter either a statement of intent or a letter of recommendation:")

    #Hide footer "made with streamlit"
    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    # Use model
    if st.button("Let's check this text!"):
        if text.strip() == "":
            st.error("Please enter some text")
        else:
            with st.spinner("Wait for the magic 🪄🔮"):
                # Use model
                if option in ("Naive Bayes", "Logistic Regression"):
                    prediction, predict_proba = nb_lr(model, text)
                    st.session_state["sklearn"] = True
                else:
                    prediction, predict_proba = torch_pred(tokenizer, model, text)
                    st.session_state["torch"] = True

            # Store the result in session state
            st.session_state["color_pred"] = "blue" if prediction == 0 else "red"
            prediction = pred_str(prediction)
            st.session_state["prediction"] = prediction
            st.session_state["predict_proba"] = predict_proba
            st.session_state["text"] = text
            
            # Print result
            st.markdown(f"I think this text is: **:{st.session_state['color_pred']}[{st.session_state['prediction']}]**",
                    help=f"Prediction probability: {st.session_state['predict_proba'] * 100}%")

    elif "prediction" in st.session_state:
        # Display the stored result if available        
        st.markdown(f"I think this text is: **:{st.session_state['color_pred']}[{st.session_state['prediction']}]**",
                    help=f"Prediction probability: {st.session_state['predict_proba'] * 100}%")

    if st.button("Model Explanation"):
        # Check if there's text in the session state
        if "text" in st.session_state and "prediction" in st.session_state:
           
            if option in ("Naive Bayes", "Logistic Regression"):
                 with st.spinner('Wait for it 💭...'):
                    explainer = TextExplainer(sampler=MaskingTextSampler())
                    explainer.fit(st.session_state["text"], model.predict_proba)
                    html = eli5.format_as_html(explainer.explain_prediction(target_names=["Human", "AI"]))
            else:
                with st.spinner('Wait for it 💭... BERT-based model explanations take around 4-10 minutes. In case you want to abort, refresh the page.'):
                # TORCH EXPLAINER PRED FUNC (USES logits)
                    def f(x):
                        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x])#.cuda()
                        outputs = model(tv).detach().cpu().numpy()
                        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
                        val = scipy.special.logit(scores[:,1]) # use one vs rest logit units
                        return val
                    # build an explainer using a token masker
                    explainer = shap.Explainer(f, tokenizer)
                    shap_values = explainer([st.session_state["text"]], fixed_context=1)
                    html = shap.plots.text(shap_values, display=False)
            # Render HTML
            st.components.v1.html(html, height=500, scrolling = True)
        else:
            st.error("Please enter some text and click 'Let's check!' before requesting an explanation.") 
            
if __name__ == "__main__":
    main()
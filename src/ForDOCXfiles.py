def main():
    """
    This function is used to read the docx files in the directory and save the text in a csv file.
    """
    import os
    import pandas as pd
    import docx

    dir = "EDGE 6101-Final Paper-Summarized by Zhang"
    list_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.docx'):        

                file_path = os.path.join(root, file)
                doc = docx.Document(file_path)
                doc_str = ""
                for parag in doc.paragraphs:
                    doc_str += parag.text
                list_files.append(doc_str)
    word_file = pd.Series(list_files, name="Word_files")

    word_file.to_csv("docs.csv",index=False)

if __name__ == "__main__":
    main()
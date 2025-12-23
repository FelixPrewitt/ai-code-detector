def clean_code(code_list): #creates function to clean code snippets 
    cleaned = []
    for snippet in code_list:
        snippet = snippet.strip()
        snippet = snippet.replace("\t", " ")
        cleaned.append(snippet)
    return cleaned


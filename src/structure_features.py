import re 

def extract_structure_features(code_snippets):
    features = []

    for code in code_snippets:
        num_lines = code.count('\n') + 1

        num_loops = len(re.findall(r"\bfor\b|\bwhile\b", code))
        num_conditionals = len(re.findall(r"\bif\b|\belif\b|\belse\b", code))
        num_returns = len(re.findall(r"\breturn\b", code))

        builtin_calls = len(re.findall(
            r"\b(len|sum|max|min|set|dict|list|sorted|any|all)\b", code))
        
        total_tokens = max(len(code.split()), 1)

        builtin_ratio = builtin_calls / total_tokens

        features.append([
            num_lines,
            num_loops,
            num_conditionals,
            num_returns,
            builtin_ratio
        ])

    return features

from predict import predict_code

code = """
def safe_div(a, b):
    return a / b if b != 0 else 0
"""

print(predict_code(code))

"""
test_predictions.py
===================
Quick smoke-test: loads the saved pipeline and runs 6 labelled predictions.
Run:  python test_predictions.py
"""
import pathlib
import sys
import joblib
# Import so the pipeline's preprocessor reference resolves correctly
from text_utils import clean_text  # noqa: F401

MODEL_PATH = pathlib.Path("model/spam_pipeline.joblib")

if not MODEL_PATH.exists():
    sys.exit("[ERROR] model/spam_pipeline.joblib not found. Run train_model.py first.")

pipeline = joblib.load(MODEL_PATH)

tests = [
    ("Congratulations! You won a 1000 prize. Call 08700621170 now to claim!", "SPAM"),
    ("FREE entry 2 win FA Cup final tkts! Text FA to 87121 now. T and C apply.", "SPAM"),
    ("URGENT! Your mobile has won a 2000 bonus. Call 09061743810 to claim.", "SPAM"),
    ("Hey, are you free tonight? Want to grab dinner?", "HAM"),
    ("Team standup at 10am tomorrow, please dont miss it.", "HAM"),
    ("Can you pick up milk and bread on the way home?", "HAM"),
]

header = f"{'#':<3} {'Expected':<8} {'Predicted':<10} {'Spam%':>7}  Message"
print(header)
print("-" * 90)

all_pass = True
for i, (msg, expected) in enumerate(tests, 1):
    proba = pipeline.predict_proba([msg])[0]
    pred  = "SPAM" if proba[1] > 0.5 else "HAM"
    ok    = "PASS" if pred == expected else "FAIL"
    if pred != expected:
        all_pass = False
    snippet = msg[:60] + ("..." if len(msg) > 60 else "")
    print(f"{i:<3} {expected:<8} {pred:<10} {proba[1]*100:>6.1f}%  {ok}  {snippet}")

print()
if all_pass:
    print("All 6 predictions correct.")
else:
    print("Some predictions did not match expected labels — check the output above.")

sys.exit(0 if all_pass else 1)

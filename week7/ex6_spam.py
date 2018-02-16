
from func import processEmail

def main():
    ## Part 1: Email Preprocessing
    print("\nPreprocessing sample email (emailSample1.txt)")

    with open("./emailSample1.txt") as f:
        file_contents = f.read()
        word_indices = processEmail(file_contents)

    ## Part 2: Feature Extraction

    ## Part 3: Train Linear SVM for Spam Classification

    ## Part 4: Test Spam Classification 

    ## Part 5: Top Predictors of Spam

    ## Part 6: Try Your Own Emails




if __name__ == "__main__":
    main()
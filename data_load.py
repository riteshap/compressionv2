

class DataLoad:
    def __init__(self):
        pass

    def load_nq_data(self, a_sample: dict) -> (list[str], str, list[list[str]]):
        document_text = a_sample["document_text"]
        annotations = a_sample["annotations"]
        question_text = a_sample["question_text"]
        tokens = document_text.split(" ")
        answers = []
        for anno in annotations:
            for s_a in anno["short_answers"]:
                answers.append(tokens[s_a["start_token"]: s_a["end_token"]])
            if anno["yes_no_answer"] == "YES":
                answers.append("yes")

        return [document_text], question_text, answers

    def load_trivia_data(self, a_sample: dict) -> (list[str], str, list[list[str]]):
        question_text = a_sample["question"]
        answers = a_sample["answers"]
        document_text = []
        for a in a_sample["document_paths"]:
            doc_file = open(a["name"], "r", encoding="utf-8")
            document_text.append(doc_file.read())
        return document_text, question_text, answers



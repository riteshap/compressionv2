import json
from compress_v2 import CompressV2
from data_load import DataLoad

data_set_type = "nq"
rate = 0.1
data_load = DataLoad()
compress_v2 = CompressV2()
compress_v2.prepare_load()
file_path = "./data_test/nq_test_data.json"
file = open(file_path, "r", encoding="utf-8")
for i, line in enumerate(file.readlines()):
    a_sample = json.loads(line)
    if data_set_type == "nq":
        document_text, question_text, answers = data_load.load_nq_data(a_sample)
    elif data_set_type == "trivia":
        document_text, question_text, answers = data_load.load_trivia_data(a_sample)
    else:
        raise Exception("error")

    _, ret_content, _, _, = compress_v2.compress_v2(document_text, question_text, rate=rate)

    print(ret_content)


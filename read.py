MODES = ["low", "medium", "high"]

def read_languages_infile(infile):
    answer = set()
    with open(infile, "r", encoding="utf8") as fin:
        for elem in fin.read().split():
            elem = elem.split("-", maxsplit=1)
            if len(elem) == 1:
                modes = MODES
            else:
                modes = elem[1].split(",")
            for mode in modes:
                if mode in MODES:
                    answer.add((elem[0], mode))
    answer = sorted(answer)
    return answer


def read_infile(infile):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            splitted = line.strip().split("\t")
            if len(splitted) != 3:
                continue
            answer.append(splitted)
    return answer



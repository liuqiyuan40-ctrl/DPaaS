import random, string

def randstr(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

# report formater, make sure it is compact !!!
# report are dict of filename -> list of (conf, reason) for each stage in the pipeline
# report output should be each line is a file, with its evaluation results for each stage, connected with "->"
# ignore the reason if it is not the last stage of the report
def format_report(report):
    res = ""
    for filename, stage_reports in report.items():
        res += f"{filename}: "
        for i, (conf, reason) in enumerate(stage_reports):
            res += f"{conf:.2f}"
            if i < len(stage_reports) - 1:
                res += " → "
            else:
                if reason:
                    res += f" ({reason})"
        res += "\n"
    return res
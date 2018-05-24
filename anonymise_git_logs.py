import os, sys, csv

COL_HASH = 0
COL_PARENTS = 1
COL_AUTHOR = 2
COL_AUTHOR_MAIL = 3
COL_AUTHOR_DATE = 4
COL_COMMITTER = 5
COL_COMMITTER_MAIL = 6
COL_COMMITTER_DATE = 7

def __replace__(actual, key):
    map = MAP[key]
    if map.has_key(actual):
        result = map.get(actual)
    else:
        result = "%s_%s" % (key, COUNTER[key])
        map[actual] = result
        COUNTER[key] = COUNTER[key] + 1
    return result

def replace_author_name(name):
    return __replace__(name, "author_name")


def replace_author_mail(mail):
    return __replace__(mail, "author_mail")


def replace_committer_name(name):
    return __replace__(name, "committer_name")


def replace_committer_mail(mail):
    return __replace__(mail, "committer_mail")


def run(log_dir, anonoyme_log_dir):
    global COUNTER
    global MAP
    path_name_tuples = [(os.path.join(log_dir, log_name), log_name) for log_name in os.listdir(log_dir)]
    for log_path, log_name in path_name_tuples:
        COUNTER = {'author_name': 0, 'author_mail': 0, 'committer_name': 0, 'committer_mail': 0}
        MAP = {'author_name': {}, 'author_mail': {}, 'committer_name': {}, 'committer_mail': {}}

        with open(log_path, 'rU') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            anonymised_lines = []
            poor_format = False
            for row in reader:
                if not len(row) == 8:
                    poor_format = True
                    break

                row[COL_AUTHOR] = replace_author_name(row[COL_AUTHOR])
                row[COL_AUTHOR_MAIL] = replace_author_mail(row[COL_AUTHOR_MAIL])
                row[COL_COMMITTER] = replace_committer_name(row[COL_COMMITTER])
                row[COL_COMMITTER_MAIL] = replace_committer_mail(row[COL_COMMITTER_MAIL])
                anonymised_lines.append("%s,%s,%s,%s,%s,%s,%s,%s" % (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]))
            if not poor_format:
                save_to_path = os.path.join(anonoyme_log_dir, log_name)
                with open(save_to_path, 'wb') as anonymised_log:
                    for line in anonymised_lines:
                        anonymised_log.write(line+"\n")
            print


assert len(sys.argv) == 3
LOGS_DIRECTORY = os.path.expanduser(sys.argv[1])
ANONYMISED_LOGS_DIR = os.path.expanduser(sys.argv[2])
assert os.path.isdir(LOGS_DIRECTORY)
assert os.path.isdir(ANONYMISED_LOGS_DIR)
run(log_dir=LOGS_DIRECTORY, anonoyme_log_dir=ANONYMISED_LOGS_DIR)

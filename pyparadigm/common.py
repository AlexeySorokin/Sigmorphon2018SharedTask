"""
Файл, содержащий описания морфологических категорий для разных языков
и функции работы с ними
"""

from itertools import product

# константы
LEMMA_KEY = "lemma"

# коды языков
LANGUAGE_CODES = ["RU", "LA", "RU_verbs", "FI"]

# категории для отдельных языков
# русский, существительные
ru_cases = ['им', 'род', 'дат', 'вин', 'тв', 'пр']
ru_numbers = ['ед', 'мн']
# русский, глаголы
ru_verb_cats = ['инфинитив', 'наст,1л,ед', 'наст,2л,ед', 'наст,3л,ед',
                'наст,1л,мн', 'наст,2л,мн', 'наст,3л,мн',  'пр,муж',
                'пр,жен', 'пр,ср', 'пр,мн', 'повел,ед', 'повел,мн']
# ru_verb_cats = ['инфинитив', 'наст,1л,ед', 'наст,2л,ед', 'наст,3л,ед',
#                 'наст,1л,мн', 'наст,2л,мн', 'наст,3л,мн',  'пр,муж',
#                 'пр,жен', 'пр,ср', 'пр,мн', 'прич,действ,наст', 'прич,действ,пр',
#                 'деепр,наст', 'деепр,прош', 'прич,страд,наст', 'прич,страд,пр']
# латинский
la_cases = ['ном', 'ген', 'дат', 'акк', 'абл', 'вок']
la_numbers = ['ед', 'мн']
# финский
fi_cases = ['nominative', 'genitive', 'partitive', 'inessive', 'elative', 'illative',
            'adessive', 'ablative', 'allative', 'essive', 'translative', 'instructive',
            'abessive', 'comitative']
fi_numbers = ['singular', 'plural']


def get_categories_marks(language):
    """
    Возвращает список категорий в зависимости от языка
    """
    if language not in LANGUAGE_CODES:
        raise ValueError("Only {0} codes are supported.".format(", ".join(LANGUAGE_CODES)))
    if language == "RU":
        marks = list(map(tuple, product(ru_cases, ru_numbers)))
    elif language == 'RU_verbs':
        marks = ru_verb_cats
    elif language == "LA":
        marks = list(map(tuple, product(la_cases, la_numbers)))
    elif language == "FI":
        marks = list(map(tuple, product(fi_cases, fi_numbers)))
    return marks

def get_possible_lemma_marks(language):
    '''
    Возвращает список категорий, в которых может содержаться лемма,
    в порядке убывания их приоритета
    '''
    if language not in LANGUAGE_CODES:
        raise ValueError("Only {0} codes are supported.".format(", ".join(LANGUAGE_CODES)))
    if language == "RU":
        answer = [('им', 'ед'), ('им', 'мн')]
    elif language == "LA":
        answer = [('ном', 'ед'), ('ном', 'мн')]
    return answer
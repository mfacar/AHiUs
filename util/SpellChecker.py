import hunspell

dic_any = hunspell.Hunspell("es_ANY")
dic_es = hunspell.Hunspell("es_ES", "es_ES")
dic_en = hunspell.Hunspell("en_US", "en_US")

res = dic_any.spell("análisis")


def check_spell(words):
    count = 0
    count_en = 0
    if len(words) == 0:
        return 1
    for word in words:
        correct_es = dic_any.spell(word) or dic_es.spell(word)
        count = count + 1 if correct_es else count
        if not correct_es:
            correct_en = dic_any.spell(word)
            count_en = count_en + 1 if correct_en else count
    return count/len(words),  count_en


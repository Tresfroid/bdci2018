import pandas as pd
import jieba.analyse
import re
from pyltp import SentenceSplitter


def extract(df, theme_words):
    sentence = "\n".join(get_theme_sentence(df, theme_words))
    words = jieba.analyse.extract_tags(sentence, allowPOS=("a"))
    return words


def contains(sent, words):
    for word in words:
        if word in sent:
            return True
    return False


def get_theme_sentence(df, theme_words):
    sentence = []
    for content in df["content"]:
        content = re.sub("[, ,]", "o", content)
        for sub in SentenceSplitter.split(content):
            if contains(sub, theme_words):
                sentence.append(sub)
    return sentence


def get_prefix(df, word, topK):
    jieba.initialize()
    jieba.add_word(word)
    df_temp = df[df["content"].str.contains(word)]
    prefix = df_temp["content"].apply(lambda x: "/t".join(jieba.cut_for_search(x))).str.extract(
        "([\u4e00-\u9fa5]+)/t{}".format(word), expand=False).value_counts().head(topK).index.tolist()
    return [pre + word for pre in prefix]


def gen_senti_dict(df, fname):
    themeDict = {}
    df_theme = pd.read_excel("../Data/data_processing/subj.xlsx")
    cols = df_theme.columns.tolist()
    for col in cols:
        themeDict[col] = df_theme[col].dropna().tolist()

    sentidict = {}
    subjects = df["subject"].unique().tolist()
    for subject in subjects:
        temp_dict = {}
        df_pos = df[(df["subject"] == subject) & (df["sentiment_value"] == 1)]
        df_neg = df[(df["subject"] == subject) & (df["sentiment_value"] == -1)]
        pos_words = list(set(
            list(filter(lambda x: len(x) < 5, df_pos["sentiment_word"].dropna().unique().tolist())) + extract(df_pos,
                                                                                                              themeDict[
                                                                                                                  subject])))
        neg_words = list(set(
            list(filter(lambda x: len(x) < 5, df_neg["sentiment_word"].dropna().unique().tolist())) + extract(df_neg,
                                                                                                              themeDict[
                                                                                                                  subject])))
        share_words = [word for word in pos_words if word in neg_words]
        if share_words:
            for word in share_words:
                pos_sub_words = get_prefix(df_pos, word, 1)
                neg_sub_words = get_prefix(df_neg, word, 1)
                pos_words.remove(word)
                pos_words.extend(pos_sub_words)
                neg_words.remove(word)
                neg_words.extend(neg_sub_words)
        temp_dict["pos"] = pos_words
        temp_dict["neg"] = neg_words
        sentidict[subject] = temp_dict

    with open(fname, "w") as f:
        for subject in subjects:
            f.write("{}: \n".format(subject))
            f.write("positive words: \n")
            f.write(",".join(sentidict[subject]["pos"]) + "\n")
            f.write("negtive words: \n")
            f.write(",".join(sentidict[subject]["neg"]) + "\n")
            f.write("-" * 100 + "\n")


if __name__ == "__main__":
    jieba.load_userdict("../Data/data_processing/car.dict")
    df = pd.read_csv("../Data/data_processing/train_2.csv")
    gen_senti_dict(df.drop_duplicates(keep=False), "../Data/data_processing/senti_dict.txt")

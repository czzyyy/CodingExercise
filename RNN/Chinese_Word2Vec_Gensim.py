# https://radimrehurek.com/gensim/models/word2vec.html 使用
import gensim.models.word2vec as word2vec


def train(filname, savename):
    sentences = word2vec.LineSentence(filname)
    model = word2vec.Word2Vec(sentences, size=200)
    model.save(savename)


if __name__ == '__main__':
    segmentname = 'F:/python_code/text/news_sohusite_xml_full/news_sohusite_segment.dat'
    modelname = 'F:/python_code/text/news_sohusite_xml_full/news_sohusite_model.bin'
    # train(segmentname, modlename)
    model = word2vec.Word2Vec.load(modelname)
    print(model.similarity(u"百姓", u"人民"))
    y = model.most_similar(u"北京", topn=20)  # 20个最相关的
    print(u"和【北京】最相关的词有：\n")
    for item in y:
        print(item[0], item[1])
    #print(u"【北京】的向量是：\n", model[u"北京"])
    y2 = model.wv.most_similar(positive=[u"男人", u"丈夫"], negative=[u"女人"], topn=2)
    print(u"距离计算【男人】【丈夫】，【女人】：\n", y2)


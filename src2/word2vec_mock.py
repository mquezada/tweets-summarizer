

class w2v(dict):
    def most_similar(self, word, topn=3):
        res = []
        for i in range(topn):
            res.append((word + "_%d" % i, 0.85))
        return res

    def __contains__(self, item):
        return True
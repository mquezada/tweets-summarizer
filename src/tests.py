import unittest
from process_text import clean_url


class TestProcessText(unittest.TestCase):

    def test_clean_url(self):
        test_urls = ['http://www.python.org/doc/?a=1&b=2',
                     'https://www.yahoo.com/news/pistorius-vomits-during-graphic-testimony-130018791--spt.html?utm_source=dlvr.it&utm_medium=twitter&ref=gs',
                     'HTTP://www.Python.org/doc/?q=qsd&r=asd#',
                     'http://a.b.c:8000/',
                     '//remotesite.com/image1.jpg']

        corr_urls = ['http://www.python.org/doc/',
                     'https://www.yahoo.com/news/pistorius-vomits-during-graphic-testimony-130018791--spt.html',
                     'http://www.Python.org/doc/',
                     'http://a.b.c:8000/',
                     '//remotesite.com/image1.jpg']

        for test, correct in zip(test_urls, corr_urls):
            self.assertEqual(clean_url(test), correct)


if __name__ == '__main__':
    unittest.main()
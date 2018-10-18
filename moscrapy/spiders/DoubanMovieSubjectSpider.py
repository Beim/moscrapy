from scrapy.spiders import Spider
from moscrapy.items import DoubanMovieSubjectItem
from scrapy import Request
from bs4 import BeautifulSoup
import json
from moscrapy.services.DoubanApi import DoubanApi
from moscrapy.db.dbpeewee import Tag, db

class DoubanSubjectSpider(Spider):
    name = 'douban_movie_subject_spider'
    doubanApi = DoubanApi()

    def start_requests(self):
        tag_list = Tag.select().where(Tag.visit == 0).limit(20)
        for tagres in tag_list:
            item = DoubanMovieSubjectItem()
            item['from_tag'] = tagres.name
            request = Request(self.doubanApi.get_search_movies_by_tag_url(tagres.name, page_start=0), callback=self.parse, priority=0)
            request.meta['item'] = item
            request.meta['tagres'] = tagres
            request.meta['page_start'] = 0
            yield request

    def parse(self, response):
        item = response.meta['item']
        tagres = response.meta['tagres']
        page_start = response.meta['page_start']
        res = json.loads(response.body)
        subjects = res['subjects']
        if len(subjects) == 0:
            tagres.visit = 1
            tagres.save()
        else:
            # subjects
            for movie in subjects:
                item['douban_id'] = movie['id']
                yield item
                # nitem = item.copy()
                # movie_id = movie['id']
                # nitem['movie_info'] = {'doubanId': movie_id}
                # request = Request(self.doubanApi.get_subject_by_id_url(movie_id), callback=self.parse_movie_info, priority=1)
                # request.meta['item'] = nitem
                # yield request
            # next page
            next_page_start = page_start + 20
            nitem = item.copy()
            request = Request(self.doubanApi.get_search_movies_by_tag_url(tagres.name, page_start=next_page_start), callback=self.parse, priority=0)
            request.meta['item'] = nitem
            request.meta['tagres'] = tagres
            request.meta['page_start'] = next_page_start
            yield request
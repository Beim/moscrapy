from scrapy.spiders import Spider
from moscrapy.items import DoubanDirectorItem
from scrapy import Request
from bs4 import BeautifulSoup
import json
from moscrapy.services.DoubanApi import DoubanApi
from moscrapy.db.dbpeewee import Director

class DoubanDirectorInfoSpider(Spider):
    name = 'douban_director_info_spider'
    doubanApi = DoubanApi()

    def start_requests(self):
        director_list = Director.select().where(Director.name  == None).limit(100)
        for director in director_list:
            item = DoubanDirectorItem()
            item['id'] = director.id
            item['doubanId'] = director.doubanId
            request = Request(self.doubanApi.get_celebrity_by_id_url(director.doubanId), callback=self.parse, priority=0)
            request.meta['item'] = item
            yield request

    def parse(self, response):
        item = response.meta['item']
        actor_info = json.loads(response.body.decode('utf-8'))
        item['name'] = actor_info['name']
        item['gender'] = actor_info['gender']
        item['foreignName'] = actor_info['name_en']
        item['bornPlace'] = actor_info['born_place']
        item['dataFrom'] = response.url
        yield item
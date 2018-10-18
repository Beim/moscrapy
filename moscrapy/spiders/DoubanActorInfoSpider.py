from scrapy.spiders import Spider
from moscrapy.items import DoubanActorItem
from scrapy import Request
from bs4 import BeautifulSoup
import json
from moscrapy.services.DoubanApi import DoubanApi
from moscrapy.db.dbpeewee import Actor

class DoubanActorInfoSpider(Spider):
    name = 'douban_actor_info_spider'
    doubanApi = DoubanApi()

    def start_requests(self):
        actor_list = Actor.select().where(Actor.name  == None).limit(100)
        for actor in actor_list:
            item = DoubanActorItem()
            item['id'] = actor.id
            item['doubanId'] = actor.doubanId
            request = Request(self.doubanApi.get_celebrity_by_id_url(actor.doubanId), callback=self.parse, priority=0)
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
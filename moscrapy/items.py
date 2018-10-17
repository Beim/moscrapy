# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MoscrapyItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

class DoubanItem(scrapy.Item):
    from_tag = scrapy.Field()
    movie_info = scrapy.Field()
    actor_info = scrapy.Field()
    director_info = scrapy.Field()
    genre_info = scrapy.Field()
    resource_info = scrapy.Field()

from scrapy.spiders import Spider
from moscrapy.items import DoubanMovieItem
from scrapy import Request
from bs4 import BeautifulSoup
import json
from moscrapy.services.DoubanApi import DoubanApi
from moscrapy.db.dbpeewee import Tag, db, Movie

class DoubanMovieInfoSpider(Spider):
    name = 'douban_movie_info_spider'
    doubanApi = DoubanApi()

    def start_requests(self):
        movie_list = Movie.select().where(Movie.name == None).limit(20)
        for movie in movie_list:
            item = DoubanMovieItem()
            item['id'] = movie.id
            item['doubanId'] = movie.doubanId
            request = Request(self.doubanApi.get_subject_by_id_url(movie.doubanId), callback=self.parse, priority=0)
            request.meta['item'] = item
            yield request


    def parse(self, response):
        item = response.meta['item']
        # from scrapy.shell import inspect_response
        # inspect_response(response, self)
        movie_info = json.loads(response.body.decode('utf-8'))
        item['dataFrom'] = response.url
        item['name'] = movie_info['title']
        item['year'] = movie_info['year']
        item['originalName'] = movie_info['original_title']
        item['place'] = ' '.join(movie_info['countries'])
        item['description'] = movie_info['summary']
        item['genre_info'] = movie_info['genres']
        movie_url = movie_info['alt']

        actors_info = []
        for actor in movie_info['casts']:
            if actor['id'] != None:
                actors_info.append({'doubanId': actor['id']})
        item['actor_info'] = actors_info

        directors_info = []
        for director in movie_info['directors']:
            if director['id'] != None:
                directors_info.append({'doubanId': director['id']})
        item['director_info'] = directors_info

        request = Request(movie_url, callback=self.parse_movie_website, priority=2)
        request.meta['item'] = item
        yield request

    def parse_movie_website(self, response):
        item = response.meta['item']
        soup = BeautifulSoup(response.body, 'lxml')
        try:
            item['language'] = soup.find_all('div', id='info')[0].find_all('span', text='语言:')[0].next_sibling.strip()
        except:
            item['language'] = None

        # extract tags
        tags = []
        try:
            for a_tag in soup.find('div', class_='tags-body').find_all('a'):
                tag = a_tag.text.strip()
                tags.append(tag)
        except:
            pass
        inserted_tags = []
        with db.atomic():
            for tag in tags:
                try:
                    Tag.create(name=tag)
                    inserted_tags.append(tag)
                except:
                    pass

        # extract play info
        item['resource_info'] = []
        if soup.find('ul', class_='bs') != None:
            try:
                src_list = soup.find('ul', class_='bs').find_all('li')
                for src in src_list:
                    price = src.find('span').text.strip()
                    platform = src.find('a')['data-cn']
                    url = src.find('a')['href']
                    item['resource_info'].append({
                        'price': price, 'platform': platform, 'url': url, 'dataFrom': response.url
                    })
            except:
                pass

        yield item
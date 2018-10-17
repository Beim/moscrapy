from scrapy.spiders import Spider
from moscrapy.items import DoubanItem
from scrapy import Request
from bs4 import BeautifulSoup
import json
from moscrapy.services.DoubanApi import DoubanApi
from moscrapy.util.Util import get_random_user_agent
from moscrapy.db.dbpeewee import Tag, db

class DoubanSpider(Spider):
    name = 'douban_movie_spider'
    doubanApi = DoubanApi()

    def start_requests(self):
        res = Tag.select().where(Tag.visit == 0).limit(1)
        if len(res) == 0: return
        tagres = res[0]
        item = DoubanItem()
        item['from_tag'] = tagres.name
        request = Request(self.doubanApi.get_search_movies_by_tag_url(tagres.name, page_start=0), callback=self.parse)
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
                nitem = item.copy()
                movie_id = movie['id']
                nitem['movie_info'] = {'doubanId': movie_id}
                request = Request(self.doubanApi.get_subject_by_id_url(movie_id), callback=self.parse_movie_info)
                request.meta['item'] = nitem
                yield request
            # next page
            next_page_start = page_start + 20
            nitem = item.copy()
            request = Request(self.doubanApi.get_search_movies_by_tag_url(tagres.name, page_start=next_page_start), callback=self.parse)
            request.meta['item'] = nitem
            request.meta['tagres'] = tagres
            request.meta['page_start'] = next_page_start
            yield request


    def parse_movie_info(self, response):
        item = response.meta['item']
        # from scrapy.shell import inspect_response
        # inspect_response(response, self)
        movie_info = json.loads(response.body.decode('utf-8'))
        item['movie_info']['dataFrom'] = response.url
        item['movie_info']['name'] = movie_info['title']
        item['movie_info']['year'] = movie_info['year']
        item['movie_info']['originalName'] = movie_info['original_title']
        item['movie_info']['place'] = ' '.join(movie_info['countries'])
        item['movie_info']['description'] = movie_info['summary']
        item['genre_info'] = movie_info['genres']
        movie_url = movie_info['alt']

        actors_info = []
        for actor in movie_info['casts']:
            actors_info.append({'doubanId': actor['id']})
        item['actor_info'] = actors_info

        directors_info = []
        for director in movie_info['directors']:
            directors_info.append({'doubanId': director['id']})
        item['director_info'] = directors_info

        request = Request(movie_url, callback=self.parse_movie_website)
        request.meta['item'] = item
        yield request


    def parse_movie_website(self, response):
        item = response.meta['item']
        # from scrapy.shell import inspect_response
        # inspect_response(response, self)
        soup = BeautifulSoup(response.body, 'lxml')
        item['movie_info']['language'] = soup.find_all('div', id='info')[0].find_all('span', text='语言:')[0].next_sibling.strip()
        # extract tags
        tags = []
        for a_tag in soup.find('div', class_='tags-body').find_all('a'):
            tag = a_tag.text.strip()
            tags.append(tag)
        inserted_tags = []
        with db.atomic():
            for tag in tags:
                try:
                    Tag.create(name=tag)
                    inserted_tags.append(tag)
                except:
                    pass
        # for tag in inserted_tags:
        #     new_item = DoubanItem()
        #     new_item['from_tag'] = tag
        #     request = Request(self.doubanApi.get_search_movies_by_tag_url(tag, page_start=0), callback=self.parse)
        #     request.meta['item'] = item
        #     yield request

        # extract play info
        item['resource_info'] = []
        if soup.find('ul', class_='bs') != None:
            src_list = soup.find('ul', class_='bs').find_all('li')
            for src in src_list:
                price = src.find('span').text.strip()
                platform = src.find('a')['data-cn']
                url = src.find('a')['href']
                item['resource_info'].append({
                    'price': price, 'platform': platform, 'url': url, 'dataFrom': response.url
                })

        # request actor info
        if len(item['actor_info']) > 0:
            actor_info = item['actor_info'][0]
            doubanId = actor_info['doubanId']
            request = Request(self.doubanApi.get_celebrity_by_id_url(doubanId), callback=self.parse_actor_info)
            request.meta['item'] = item
            request.meta['actor_info_index'] = 0
            yield request


    def parse_actor_info(self, response):
        item = response.meta['item']
        actor_info_index = response.meta['actor_info_index']
        # from scrapy.shell import inspect_response
        # inspect_response(response, self)
        actor_info = json.loads(response.body.decode('utf-8'))
        curr_actor = item['actor_info'][actor_info_index]
        curr_actor['name'] = actor_info['name']
        curr_actor['gender'] = actor_info['gender']
        curr_actor['foreignName'] = actor_info['name_en']
        curr_actor['bornPlace'] = actor_info['born_place']
        curr_actor['dataFrom'] = response.url

        actor_info_index = actor_info_index + 1
        if actor_info_index < len(item['actor_info']):
            doubanId = item['actor_info'][actor_info_index]['doubanId']
            request = Request(self.doubanApi.get_celebrity_by_id_url(doubanId), callback=self.parse_actor_info)
            request.meta['item'] = item
            request.meta['actor_info_index'] = actor_info_index
            yield request
        else:
            if len(item['director_info']) > 0:
                director_info = item['director_info'][0]
                doubanId = director_info['doubanId']
                request = Request(self.doubanApi.get_celebrity_by_id_url(doubanId), callback=self.parse_director_info)
                request.meta['item'] = item
                request.meta['director_info_index'] = 0
                yield request

    def parse_director_info(self, response):
        item = response.meta['item']
        director_info_index = response.meta['director_info_index']
        # from scrapy.shell import inspect_response
        # inspect_response(response, self)
        director_info = json.loads(response.body.decode('utf-8'))
        curr_director = item['director_info'][director_info_index]
        curr_director['name'] = director_info['name']
        curr_director['gender'] = director_info['gender']
        curr_director['foreignName'] = director_info['name_en']
        curr_director['bornPlace'] = director_info['born_place']
        curr_director['dataFrom'] = response.url

        director_info_index = director_info_index + 1
        if director_info_index < len(item['director_info']):
            doubanId = item['director_info'][director_info_index]['doubanId']
            request = Request(self.doubanApi.get_celebrity_by_id_url(doubanId), callback=self.parse_director_info)
            request.meta['item'] = item
            request.meta['director_info_index'] = director_info_index
            yield request
        else:
            yield item

            # start_request
            res = Tag.select().where(Tag.visit == 0).limit(1)
            if len(res) == 0: return
            tagres = res[0]
            item = DoubanItem()
            item['from_tag'] = tagres.name
            request = Request(self.doubanApi.get_search_movies_by_tag_url(tagres.name, page_start=0), callback=self.parse)
            request.meta['item'] = item
            request.meta['tagres'] = tagres
            request.meta['page_start'] = 0
            yield request



import requests
from moscrapy.db.dbpeewee import Movie, Genre, Actor, Director, MovieToActor, MovieToDirector, MovieToGenre, Resource
# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


class MoscrapyPipeline(object):
    def process_item(self, item, spider):
        print(item)
        return item

class TestSpiderPipeline(object):
    def process_item(self, item, spider):
        if spider.name != 'test_spider': return item
        # with open('./test.txt', 'w') as f:
        #     f.write(item['from_tag'])
        return item

class DoubanMovieSpiderPipeline(object):
    def process_item(self, item, spider):
        if spider.name != 'douban_movie_spider': return item

        movie_info = item['movie_info']
        movie_record, is_new = Movie.insert_movie(movie_info)
        print('insert movie')
        print(movie_info)

        if is_new:
            resource_info_list = item['resource_info']
            for resource_info in resource_info_list:
                resource_info['movieId'] = movie_record.id
                Resource.create(**resource_info)
                print('insert resource')
                print(resource_info)

            genre_info_list = item['genre_info']
            for genre_name in genre_info_list:
                genre_info = {'name': genre_name}
                genre_record = Genre.insert_genre(genre_info)
                print('insert genre')
                print(genre_info)

                movie_to_genre_info = {'movieId': movie_record.id, 'genreId': genre_record.id}
                MovieToGenre.insert_movie_to_genre(movie_to_genre_info)

            actor_info_list = item['actor_info']
            for actor_info in actor_info_list:
                actor_record = Actor.insert_actor(actor_info)
                print('insert actor')
                print(actor_info)

                movie_to_actor_info = {'movieId': movie_record.id, 'actorId': actor_record.id}
                MovieToActor.insert_movie_to_actor(movie_to_actor_info)

            director_info_list = item['director_info']
            for director_info in director_info_list:
                director_record = Director.insert_director(director_info)
                print('insert director')
                print(director_info)

                movie_to_director_info = {'movieId': movie_record.id, 'directorId': director_record.id}
                MovieToDirector.insert_movie_to_director(movie_to_director_info)

        return item



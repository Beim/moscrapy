from peewee import *

db = MySQLDatabase("modemo", host="119.29.160.85", user="root",passwd="112223334")

class BaseModel(Model):
    class Meta:
        database = db

class Actor(BaseModel):
    id = IntegerField()
    name = CharField()
    doubanId = IntegerField()
    gender = CharField()
    foreignName = CharField()
    bornPlace = CharField()
    dataFrom = CharField()

    def insert_actor(actor_info):
        try:
            actor_record = Actor.create(**actor_info)
        except IntegrityError:
            actor_record = Actor.select().where(Actor.doubanId == actor_info['doubanId']).get()
        return actor_record

class Director(BaseModel):
    id = IntegerField()
    name = CharField()
    doubanId = IntegerField()
    gender = CharField()
    foreignName = CharField()
    bornPlace = CharField()
    dataFrom = CharField()

    def insert_director(director_info):
        try:
            director_record = Director.create(**director_info)
        except IntegrityError:
            director_record = Director.select().where(Director.doubanId == director_info['doubanId']).get()
        return director_record

class Genre(BaseModel):
    id = IntegerField()
    name = CharField()

    def insert_genre(genre_info):
        try:
            genre_record = Genre.create(**genre_info)
        except IntegrityError:
            genre_record = Genre.select().where(Genre.name == genre_info['name']).get()
        return genre_record

class Movie(BaseModel):
    id = IntegerField()
    name = CharField()
    language = CharField()
    place = CharField()
    description = TextField()
    doubanId = IntegerField()
    dataFrom = CharField()
    year = IntegerField()
    originalName = CharField()

    def insert_movie(movie_info):
        try:
            movie_record = Movie.create(**movie_info)
            is_new = 1
        except IntegrityError:
            movie_record = Movie.select().where(Movie.name == movie_info['name']).get()
            is_new = 0
        return movie_record, is_new

class MovieToActor(BaseModel):
    movieId = IntegerField()
    actorId = IntegerField()

    def insert_movie_to_actor(info):
        try:
            record = MovieToActor.create(**info)
        except:
            record = None
        return record

class MovieToDirector(BaseModel):
    movieId = IntegerField()
    actorId = IntegerField()

    def insert_movie_to_director(info):
        try:
            record = MovieToDirector.create(**info)
        except:
            record = None
        return record

class MovieToGenre(BaseModel):
    movieId = IntegerField()
    genreId = IntegerField()

    def insert_movie_to_genre(info):
        try:
            record = MovieToGenre.create(**info)
        except:
            record = None
        return record

class Resource(BaseModel):
    id = IntegerField()
    url = CharField()
    platform = CharField()
    price = CharField()
    doubanId = IntegerField()
    dataFrom = CharField()
    movieId = IntegerField()



class Tag(BaseModel):
    id = IntegerField()
    name = CharField()
    visit = CharField()

class Test(BaseModel):
    id = IntegerField()
    name = CharField()
    visit = CharField()

    def insert_test(test_info):
        print(test_info)
        try:
            test_record = Test.create(**test_info)
        except IntegrityError:
            test_record = Test.select().where(Test.name == test_info['name']).get()
        return test_record

if __name__ == '__main__':
    pass
    # g = Genre.create(name='g5')
    # print(g)

    # for i in range(5):
    #     Test.create(name=str(i)+'mx', visit=0)

    # res = Test.insert_test({'name': '999'})
    # print(res)
    # try:
    #     res = Test.create(name='1')
    #     print(res)
    # except IntegrityError as e:
    #     print(e)


    # res = Test.select()
    # print(len(res))
    # print(res.name)
    # for r in res:
    #     print(r)
    #     r.visit = 0
    #     r.save()

    # names = ['a4', 'a5', 'a3']
    # with db.atomic():
    #     for name in names:
    #         try:
    #             Genre.create(name=name)
    #         except:
    #             print('%s duplicate' % name)
    # g.save()
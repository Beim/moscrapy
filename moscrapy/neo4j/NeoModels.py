from py2neo.ogm import GraphObject, Property, Label, RelatedTo

class Movie(GraphObject):
    rid = Property()
    name = Property()
    language = Property()
    place = Property()
    description = Property()
    doubanId = Property()
    dataFrom = Property()
    year = Property()
    originalName = Property()

    hasActor = RelatedTo('Actor')
    hasDirector = RelatedTo('Director')
    hasGenre = RelatedTo('Genre')
    hasResource = RelatedTo('Resource')

class Genre(GraphObject):
    rid = Property()
    name = Property()

class Actor(GraphObject):
    rid = Property()
    name = Property()
    doubanId = Property()
    gender = Property()
    foreignName = Property()
    bornPlace = Property()
    dataFrom = Property()

class Director(GraphObject):
    rid = Property()
    name = Property()
    doubanId = Property()
    gender = Property()
    foreignName = Property()
    bornPlace = Property()
    dataFrom = Property()

class Resource(GraphObject):
    rid = Property()
    url = Property()
    platform = Property()
    price = Property()
    dataFrom = Property()
    movieId = Property()

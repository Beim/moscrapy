
class DoubanApi(object):

    def get_celebrity_by_id_url(self, id):
        return 'https://api.douban.com/v2/movie/celebrity/%s' % id

    def get_search_subjects_by_tag_url(self, type, tag, page_start, page_limit=20):
        return 'https://movie.douban.com/j/search_subjects?type=%s&tag=%s&page_limit=%s&page_start=%s' % (type, tag, page_limit, page_start)

    def get_search_movies_by_tag_url(self, tag, page_start, page_limit=20):
        return self.get_search_subjects_by_tag_url('movie', tag, page_start, page_limit)

    def get_subject_by_id_url(self, id):
        return 'https://api.douban.com/v2/movie/subject/%s' % id
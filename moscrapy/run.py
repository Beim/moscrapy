from scrapy import cmdline


name = 'douban_movie_top250'
name = 'kdlspider'
name = 'douban_spider'
name = 'douban_movie_spider'
# name = 'test_spider'
cmd = 'scrapy crawl {0}'.format(name)
cmdline.execute(cmd.split())
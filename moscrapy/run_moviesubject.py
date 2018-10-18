# -*- coding: utf-8 -*-
from scrapy import cmdline
import threading
import time
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import os
from multiprocessing import Process

# project需要改为你的工程名字（即settings.py所在的目录名字）
os.environ.setdefault('SCRAPY_SETTINGS_MODULE', 'moscrapy.settings')


def start_crawl():
    process = CrawlerProcess(get_project_settings())
    spider_list = process.spider_loader.list()
    # for spider_name in spider_list:
    #     process.crawl(spider_name)
    process.crawl('douban_movie_subject_spider')
    process.start()

# start_crawl()

if __name__ == '__main__':
    while True:
        p = Process(target=start_crawl)
        p.start()
        p.join()
        print('done. sleep 10s')
        time.sleep(10)

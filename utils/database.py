#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psycopg2
import re
import json

def insert_snippets(_article, cur):
    snippets = re.split("\n\n+", _article['text'])
    
    for snippet in snippets:
        cur.execute("INSERT INTO snippets (article_id, text) VALUES (%s, %s)", (_article['id'], snippet))

def insert_article(_conn_string, _article):
    conn = psycopg2.connect(_conn_string)
    cur = conn.cursor()
    cur.execute("INSERT INTO articles (id, title) VALUES (%s, %s)", (_article['id'], _article['title']))
    
    insert_snippets(_article, cur)
    
    cur.close()
    conn.commit()
    conn.close()

    

def delete_articles(conn_string):
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("delete from articles")
    cur.close()
    conn.commit()
    conn.close()

def process_file(_conn_string, _data):
    i = 0
    for line in _data:
        #print (line)
        article = json.loads(line)
        #print (article['id'])
        insert_article(_conn_string, article)
        i = i + 1
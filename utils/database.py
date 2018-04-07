#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import IntegrityError
import os, sys, re
import json
import shutil
import pandas as pd
#import conf.conn as cfg


def insert_snippets(_article, cur):
    snippets = re.split("\n\n+", _article['text'])
    
    for snippet in snippets:
        cur.execute("INSERT INTO snippets (article_id, text) VALUES (%s, %s)", (_article['id'], snippet))

def insert_article(_conn_string, _article, _file_id):
    conn = psycopg2.connect(_conn_string)
    cur = conn.cursor()

    try:
        cur.execute("INSERT INTO articles (id, title, file_id) VALUES (%s, %s, %s)", (_article['id'], _article['title'], _file_id,))
        insert_snippets(_article, cur)
    except IntegrityError:
        print('article '+str(_article['id'])+" already exists:", sys.exc_info()[0])

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

def mark_file_as_processed(_conn_string, _file_id):
    conn = psycopg2.connect(_conn_string)
    cur = conn.cursor()
    cur.execute("UPDATE files set processed = 't' where id = %s", (_file_id,))
    cur.close()
    conn.commit()
    conn.close()

def move_file(_filepath, _wikifilesdir, _processed_dir):
    new_file_name = _filepath.replace('/', '__')
    shutil.move(_wikifilesdir+'/'+_filepath, _processed_dir+'/'+new_file_name)
    

def process_file(cfg, _datafile, file_id):
    
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    i = 0
    for line in _datafile:
        #print (line)
        article = json.loads(line)
        #print (article['id'])
        insert_article(_conn_string, article, file_id)
        i = i + 1

    mark_file_as_processed(conn_string, file_id)

def select_files(cfg, _limit):
    
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("select * from files where processed = 'f' limit %s", (_limit,))
    return cur.fetchall()

def register_file (cfg, _path):
    
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO files (path, processed) VALUES (%s, %s)", (_path, 'f'))
    except IntegrityError:
        print('file '+_path+" already registered:", sys.exc_info()[0])
    cur.close()
    conn.commit()
    conn.close()


def insert_cleaned_text(cfg, _id, _tokens):
    
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    try:
        cur.execute("UPDATE snippets set cleaned_text = %s, cleaned = %s where id = %s", (' '.join(_tokens), 't', _id))
    except IntegrityError:
        print(sys.exc_info()[0])
    cur.close()
    conn.commit()
    conn.close()


def get_table_size(cfg):
    
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    sql = 'select table_name, (pg_size_pretty(pg_relation_size(quote_ident(table_name)))) '
    sql += 'from information_schema.tables '
    sql += "where table_schema = 'public' order by 2"
    cur.execute(sql)
    return cur.fetchall()

def update_words_vs_snippets(cfg, _n_snippets, n_words, snippets_size):
    
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    try:
        cur.execute("INSERT into words_vs_snippets (n_snippets, n_words, register_date, snippets_size) values (%s, %s, now(), %s)", (_n_snippets, n_words, snippets_size,))
    except IntegrityError:
        print(sys.exc_info()[0])
    cur.close()
    conn.commit()
    conn.close()


def get_current_chunk_id(cfg):
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    sql = "select max(idmax) from chunks"
    cur.execute(sql)
    result = cur.fetchall()
    
    if result[0][0] is None:
        sql = "select min(id) from snippets"
        cur.execute(sql)
        result = cur.fetchall()
    cur.close()
    conn.close()
    return result

def create_chunk(cfg, _min, _max, _size):
    conn_string = "dbname="+cfg.conn_string['dbname']+" user="+cfg.conn_string['user']+" password="+cfg.conn_string['password']
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    print (_min, _max, _size)
    cur.execute("INSERT into chunks (idmin, idmax, size, processed) values (%s, %s, %s, %s) RETURNING id", (int(_min), int(_max), int(_size), 'f',))
    id = cur.fetchone()[0]
    cur.close()
    conn.commit()
    conn.close()
    return id
    
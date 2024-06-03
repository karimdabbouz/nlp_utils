from sqlalchemy import create_engine, MetaData, Table, select, inspect, and_
import datetime
import pandas as pd


class ArticleLoader():
    '''
    This class loads news articles from my Postgres article database.
    tables needs to be a list of table names.
    start_date and end_date need to be a datetime object.
    '''
    def __init__(self, host, port, db_name, username, password, start_date, end_date, tables='all'):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.username = username
        self.password = password
        self.engine = create_engine(f'postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}?sslmode=require')
        self.metadata = MetaData()
        self.tables = inspect(self.engine).get_table_names() if tables == 'all' else tables
        self.start_date = start_date
        self.end_date = end_date
        self.raw_articles = self.load_articles()
        self.article_stats = self.compute_article_stats()


    def load_articles(self):
        '''
        Loads articles according to the given attributes and stores them as a list of tuples in raw_articles.
        '''
        articles = []
        for table in self.tables:
            table_data = Table(table, self.metadata, autoload_with=self.engine)
            query = select(table_data).where(and_(table_data.c.date_published >= self.start_date, table_data.c.date_published <= self.end_date))
            with self.engine.connect() as connection:
                response = connection.execute(query)
                articles.extend(response)
        return articles


    def compute_article_stats(self):
        '''
        Computes basic stats on articles loaded by given attributes.
        '''       
        total_num_articles = len(self.raw_articles)
        num_missing_headline = len([x for x in self.raw_articles if x.headline == None or x.headline == ''])
        num_missing_kicker = len([x for x in self.raw_articles if x.kicker == None or x.kicker == ''])
        num_missing_teaser = len([x for x in self.raw_articles if x.teaser == None or x.teaser == ''])
        num_missing_body = len([x for x in self.raw_articles if x.body == None or x.body == ''])
        num_full_data = len([x for x in self.raw_articles if
                            x.kicker != None and x.kicker != '' and
                            x.headline != None and x.headline != '' and
                            x.teaser != None and x.teaser != '' and
                            x.body != None and x.body != ''])
        return {
            'total_num_articles': total_num_articles,
            'num_full_data': num_full_data,
            'num_missing_headline': num_missing_headline,
            'num_missing_kicker': num_missing_kicker,
            'num_missing_teaser': num_missing_teaser,
            'num_missing_body': num_missing_body
        }
        return (total_num_articles, num_missing_headline, num_missing_kicker, num_missing_teaser, num_missing_body)


    def filter_articles(self, parts_of_article=['kicker', 'headline', 'teaser', 'body']):
        '''
        Takes raw_articles and returns only those parts of the article to keep for analysis.
        parts_of_article defines what part of the article to keep.
        Articles where one of the parts is not present, will not be included.
        Returns a list of tuples where the first value is the ID of the raw_articles and the second is a string with all parts concatenated.
        '''
        common_indices = []
        result = []
        df = pd.DataFrame(self.raw_articles)
        if 'kicker' in parts_of_article:
            filtered_df = df[(df['kicker'].notna()) & (df['kicker'] != '')]
            kicker_indices = list(filtered_df.index)
            common_indices.append(set(kicker_indices))
        if 'headline' in parts_of_article:
            filtered_df = df[(df['headline'].notna()) & (df['headline'] != '')]
            headline_indices = list(filtered_df.index)
            common_indices.append(set(headline_indices))
        if 'teaser' in parts_of_article:
            filtered_df = df[(df['teaser'].notna()) & (df['teaser'] != '')]
            teaser_indices = list(filtered_df.index)
            common_indices.append(set(teaser_indices))
        if 'body' in parts_of_article:
            filtered_df = df[(df['body'].notna()) & (df['body'] != '')]
            body_indices = list(filtered_df.index)
            common_indices.append(set(body_indices))
        
        if common_indices:
            common_indices = list(set.intersection(*common_indices))
            result_df = df.iloc[common_indices]
        
        result = [(f'{row.id}_{row.medium}', '') for i, row in result_df.iterrows()]
        if 'kicker' in parts_of_article:
            result = [(v[0], v[1] + ' ' + result_df.iloc[i].kicker) for i, v in enumerate(result)]
        if 'headline' in parts_of_article:
            result = [(v[0], v[1] + ' ' + result_df.iloc[i].headline) for i, v in enumerate(result)]
        if 'teaser' in parts_of_article:
            result = [(v[0], v[1] + ' ' + result_df.iloc[i].teaser) for i, v in enumerate(result)]
        if 'body' in parts_of_article:
            result = [(v[0], v[1] + ' ' + result_df.iloc[i].body) for i, v in enumerate(result)]
        return result
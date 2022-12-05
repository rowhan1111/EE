from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='c6d77cafe721436fa457335fe726f405')

c = newsapi.get_everything(q="stock", from_param='2022-07-09', to='2022-08-08', sort_by='popularity', page=2)

print(c)


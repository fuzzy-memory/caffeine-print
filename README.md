# Caffeine Print ☕️
This started as a tiny pet project to send a daily mailer of current affairs and political news.

# Overview
This script achieves the following:
1. Retrieves all political news generated in the last day using [worldnewsapi.com](https://www.worldnewsapi.com)
2. Clusters similar news stories using S-BERT sentence embedding and DBSCAN clustering. The best article is chosen based on the ranking of the news source
3. Ranks articles using a weighted score generated on the basis of the article text, source, sentiment and a weighted score calculated by taking ChatGPT's feedback on a set of metrics
4. Arranges the articles into an HTML body
5. Sends the articles to a set of recipients

<div style="text-align: center;">Made with ❤️ in Mumbai</div>
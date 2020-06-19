# [Crowdflower Search Results Relevance](https://www.kaggle.com/c/crowdflower-search-relevance)

#  Description
So many of our favorite daily activities are mediated by proprietary search algorithms. Whether you're trying to find a stream of that reality TV show on cat herding or shopping an eCommerce site for a new set of Japanese sushi knives, the relevance of search results is often responsible for your (un)happiness. **Currently, small online businesses have no good way of evaluating the performance of their search algorithms**, making it difficult for them to provide an exceptional customer experience.

The goal of this competition is to **create an open-source model that can be used to measure the relevance of search results.** In doing so, you'll be helping enable small business owners to match the experience provided by more resource rich competitors. It will also provide more established businesses a model to test against. Given the queries and resulting product descriptions from leading eCommerce sites, this competition asks you to evaluate the accuracy of their search algorithms.


## Data Description

To evaluate search relevancy, CrowdFlower has had their crowd evaluate searches from a handful of eCommerce websites. **A total of 261 search terms** were generated, and CrowdFlower put together a list of products and their corresponding search terms. Each rater in the crowd was asked to give a product search term a score of 1, 2, 3, 4, with 4 indicating the item completely satisfies the search query, and 1 indicating the item doesn't match the search term.

<img src="./img/Data Description.png" width=100%>

The challenge in this competition is to predict the relevance score given the product description and product title. To ensure that your algorithm is robust enough to handle any noisy HTML snippets in the wild real world, the data provided in the product description field is raw and contains information that is irrelevant to the product.



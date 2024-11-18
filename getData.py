import json

import steamreviews

# Download reviews for The Escapists 2
# review_dict, query_count = steamreviews.download_reviews_for_app_id("641990",
#                                                                     chosen_request_params={"language": "english"})
#
# json.dump(review_dict, open("review_dict.json", "w"))

# # Download reviews for Astroneer
# review_dict, query_count = steamreviews.download_reviews_for_app_id("361420",
#                                                                     chosen_request_params={"language": "english"})

# Download reviews for Portal with RTX
review_dict, query_count = steamreviews.download_reviews_for_app_id("2012840",
                                                                    chosen_request_params={"language": "english"})

# json.dump(review_dict, open("review_dict2.json", "w"))

print(len(review_dict["reviews"]))

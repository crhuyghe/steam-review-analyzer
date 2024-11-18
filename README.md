# steam-review-analyzer

To launch this application, simply run the file "main.py"

Doing so will launch a Gradio application that by default runs on http://127.0.0.1:7860.
The exact link is output to the console at runtime.


To use the application, type in the game title and app ID of the Steam game you wish to analyze reviews for. The app ID
is an integer divisible by 10 that can be seen on the link to the Steam store page for a game.

After submitting the game details, the application will download the review dataset. As Steam's API is rate limited, 
this may take some time. Once the entire review dataset is downloaded, it will be processed into a set of embeddings.
This can take a few minutes for games with many reviews.

After the review data is processed, it is saved into data files for easy access in the future, and the title is saved
into the dropdown menu on the interface to allow for quick selection.

Once a review set has been downloaded and selected, to begin querying the review set, type in a query in the query box
and press the button. The application will then use text embeddings to determine the similarity of the query to each of
the reviews in the dataset. An overall sentiment and relevance score is then generated based on those, and the most 
relevant reviews are retrieved to look through.

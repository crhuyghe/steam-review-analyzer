# Created by Christian Huyghe on 11/17/2024
# Launches predictor application with Gradio
import gradio as gr

from DataManager import DataManager

dm = DataManager()
dl_status = "Idle"
p_index = 0
n_index = 0

def update_status(status):
    global dl_status
    dl_status = status

def enable_widgets():
    return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

def disable_widgets():
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

def refresh_dropdown(val):
    if val not in dm.known_games.keys():
        val = -1
    return (gr.update(choices=[("New...", -1), *[(t, aid) for aid, t in dm.known_games.items()]], value=val),
            gr.update(interactive=val == -1), gr.update(interactive=val == -1))

def update_dropdown(val):
    if val == -1:
        return gr.update(value="", interactive=True), gr.update(value="", interactive=True)
    else:
        return gr.update(value=dm.known_games[val], interactive=False), gr.update(value=val, interactive=False)

def reset_reviews():
    global p_index
    global n_index
    p_index = 0
    n_index = 0
    if dm.query_response is None:
        p_val = "No relevant positive reviews found"
        n_val = "No relevant negative reviews found"
    else:
        if len(dm.query_response[2]["positive"]) > 0:
            p_val = dm.query_response[2]["positive"][0]
        else:
            p_val = "No relevant positive reviews found"
        if len(dm.query_response[2]["negative"]) > 0:
            n_val = dm.query_response[2]["negative"][0]
        else:
            n_val = "No relevant negative reviews found"


    return gr.update(value=p_val), gr.update(value=n_val)

def p_shift_reviews(val):
    global p_index
    if dm.query_response is None or len(dm.query_response[2]["positive"]) == 0:
        return gr.update(value="No relevant positive reviews found")
    else:
        p_index = (p_index + val) % len(dm.query_response[2]["positive"])
        return gr.update(value=dm.query_response[2]["positive"][p_index])

def n_shift_reviews(val):
    global n_index

    if dm.query_response is None or len(dm.query_response[2]["negative"]) == 0:
        return gr.update(value="No relevant negative reviews found")
    else:
        n_index = (n_index + val) % len(dm.query_response[2]["negative"])
        return gr.update(value=dm.query_response[2]["negative"][n_index])


with (gr.Blocks(theme="citrus", title="Steam Review Analyzer") as app):
    gr.Markdown("# Steam Review Sentiment Analyzer")
    gr.Markdown("Select known titles or download new review data")

    with gr.Row(equal_height=True):
        game_dropdown = gr.Dropdown([("New...", -1), *[(t, aid) for aid, t in dm.known_games.items()]],
                                    label="Game selection")

        game_box = gr.Textbox(placeholder="Input game title...", label="Game Title")
        appid_box = gr.Textbox(placeholder="Input app ID...", label="App ID", elem_id="app_id")
        dl_button = gr.Button("Load review data")
        download_status_box = gr.Textbox(value=lambda: dl_status, every=.5, label="Review status")

    with gr.Row(equal_height=True):
        query_box = gr.Textbox(placeholder="Input query...", label="Input query to evaluate sentiment")
        button = gr.Button("Determine Query Sentiment")

    with gr.Row(equal_height=True):
        curr_game_box = gr.Textbox(interactive=False, label="Current game", every=1,
                                   value=lambda: dm.curr_game if dm.curr_game else "No game selected")
        sentiment_box = gr.Textbox(interactive=False, label="Sentiment Score", every=1,
                                   value=lambda: dm.query_response[0] if dm.query_response else "")
        relevance_box = gr.Textbox(interactive=False, label="Relevance Score", every=1,
                                   value=lambda: dm.query_response[1] if dm.query_response else "")
        popularity_box = gr.Textbox(interactive=False, label="Game Popularity Score", every=1,
                                   value=lambda: dm.pop_score if dm.pop_score else "")

    with gr.Row(equal_height=True):
        p_back_button = gr.Button("<", scale=0)
        p_review_box = gr.Textbox(interactive=False, label="Relevant Positive Reviews",
                                  value="No relevant positive reviews found.")
        p_forward_button = gr.Button(">", scale=0)

    with gr.Row(equal_height=True):
        n_back_button = gr.Button("<", scale=0)
        n_review_box = gr.Textbox(interactive=False, label="Relevant Negative Reviews",
                                  value="No relevant negative reviews found.")
        n_forward_button = gr.Button(">", scale=0)

    dl_button.click(fn=disable_widgets, inputs=[], outputs=[game_dropdown, dl_button, button]
                    ).then(fn=lambda t, aid: dm.load_game(t, aid, update_status), inputs=[game_box, appid_box]
                           ).then(fn=refresh_dropdown, inputs=[appid_box], outputs=[game_dropdown, game_box, appid_box]
                                  ).then(fn=reset_reviews, inputs=[], outputs=[p_review_box, n_review_box]
                                       ).then(fn=enable_widgets, inputs=[], outputs=[game_dropdown, dl_button, button])
    button.click(fn=disable_widgets, inputs=[], outputs=[game_dropdown, dl_button, button]
                 ).then(fn=dm.query, inputs=[query_box], outputs=[]
                        ).then(fn=reset_reviews, inputs=[], outputs=[p_review_box, n_review_box]
                               ).then(fn=enable_widgets, inputs=[], outputs=[game_dropdown, dl_button, button])
    game_dropdown.input(fn=update_dropdown, inputs=[game_dropdown], outputs=[game_box, appid_box])
    p_back_button.click(fn=lambda: p_shift_reviews(-1), inputs=[], outputs=[p_review_box])
    p_forward_button.click(fn=lambda: p_shift_reviews(1), inputs=[], outputs=[p_review_box])
    n_back_button.click(fn=lambda: n_shift_reviews(-1), inputs=[], outputs=[n_review_box])
    n_forward_button.click(fn=lambda: n_shift_reviews(1), inputs=[], outputs=[n_review_box])



app.launch()


from flask import Flask, render_template, request, session, redirect
from rag_pipeline import run_query

app = Flask(__name__)
app.secret_key = "mysecretkey123"

@app.route("/", methods=["GET"])
def index():
    if not session.get("from_post"):
        session["history"] = []  # clear only on real refresh
    session["from_post"] = False
    session.modified = True
    return render_template("index.html", history=session.get("history", []))

@app.route("/chat", methods=["POST"])
def chat():
    query = request.form.get("query")
    if query:
        if "history" not in session:
            session["history"] = []
        response = run_query(query, session["history"])
        session["history"].append({"user": query, "bot": response})
        session["from_post"] = True
        session.modified = True
    return redirect("/")

@app.route("/clear")
def clear():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
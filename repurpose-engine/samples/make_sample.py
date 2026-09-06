"""Build samples/sample_transcript.json from a plain-text sample.

Run: python samples/make_sample.py
Gives the pipeline something realistic to chew on offline (--transcript).
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

TEXT = """
So the thing nobody tells you about pricing is that lowering your price almost never fixes a demand problem.
We tested this at my last company for eleven months.
We ran the same landing page at twenty nine dollars, at forty nine, and at ninety nine.
The ninety nine dollar page converted at one point eight percent.
The twenty nine dollar page converted at two point one percent.
So yes, more people bought the cheap one, but revenue per visitor was almost three times higher at ninety nine.
And here is the part that surprised me. Support tickets went up on the cheap plan.
The customers who paid the least asked for the most.
Refund rate on the twenty nine dollar plan was nine percent. On the ninety nine dollar plan it was under two percent.
So we were paying more to serve the people who paid us less.
I think the reason is that price is a filter. It sorts people before they ever talk to you.
When you drop your price you are not opening the door wider, you are changing who walks through it.
Now let me talk about churn, because this is where most early stage founders lose the plot.
Everyone obsesses over acquisition. Nobody reads the cancellation reasons.
We started reading every single cancellation note out loud in the Monday meeting.
Not summarized. Read out loud, word for word, by whoever shipped the feature they were complaining about.
That one ritual did more for our retention than any dashboard we ever built.
Within four months churn went from six percent monthly to three point four.
The reason is simple. When you hear a real person say your onboarding confused them, you fix it that week.
When you see churn as a number on a chart, you schedule a meeting about it in Q3.
Someone asked me recently how you know when to hire your first salesperson.
My answer is you do not hire a salesperson until you have personally closed thirty deals.
Not ten. Thirty. Because the first ten teach you the pitch and the next twenty teach you the objections.
If you hire a salesperson before that, you are asking them to invent your sales process for you.
And they will invent one that works for the product they sold at their last company, not yours.
I made this mistake. I hired a great rep from a big company in month five.
He was good. Our product was not ready for his playbook. He left in four months and it was my fault.
The other thing I would tell my younger self is to stop building features for the loudest customer.
We had one enterprise account, about eighteen percent of revenue, and they asked for everything.
We built a permissions system for them that took two engineers three months.
Nobody else used it. Zero other accounts turned it on.
That is three engineer months we could have spent on the import tool that everybody asked for quietly.
Loud requests are not the same as common requests. You have to count, not listen.
So now we tag every feature request with the account and we look at the count once a month.
It is a boring spreadsheet. It has saved us probably a year of engineering time.
Last thing. On working hours, because people ask about burnout.
I do not think the answer is fewer hours. I think the answer is fewer decisions.
The thing that burned me out was not working on Saturday. It was making forty small reversible decisions a day.
So we wrote down defaults. Default meeting length is fifteen minutes. Default answer to a new tool is no.
Default response to a customer asking for a discount is no, but here is a smaller plan.
Once the defaults existed the days got quieter, and the same amount of work stopped feeling heavy.
"""


def build():
    lines = [l.strip() for l in TEXT.strip().split("\n") if l.strip()]
    segments = []
    t = 12.0
    for line in lines:
        dur = max(3.0, len(line.split()) / 2.6)  # ~156 wpm
        segments.append({"start": round(t, 2), "end": round(t + dur, 2), "text": line})
        t += dur + 0.4
    data = {
        "text": " ".join(lines),
        "duration": round(t, 2),
        "language": "en",
        "segments": segments,
    }
    out = HERE / "sample_transcript.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (HERE / "sample_transcript.txt").write_text(data["text"], encoding="utf-8")
    print(f"wrote {out} ({len(segments)} segments, {round(t/60,1)} min)")


if __name__ == "__main__":
    build()

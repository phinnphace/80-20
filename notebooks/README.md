## The case for the radical  打  (dǎ)
While working on a prior project on VM I began to ruminate on how we (regardless of domain) train, trial, test and experiment. Without fail, establishing a 'baseline' for the aforementioned occurs under artificial, curated and manipulated conditions. Ultimately these environments are not where the very phenonoma we are testing will be used. Adding many additional hurdels to the process. What if instead of trying out our new idea in an artifical environement divorced from its intended use, we began where it will be used? Wouldn't this remove uneccessary hurdles, and force us to tackle real ones? Wouldn't the results be inherently less fragile because they are designed and built in and for where they will be implemented?


In 2020, an episode of Radiolab called "The Wubi Effect" has never left me. I just put it back on and it is so hard to press pause on... If you are not familiar with this I cannot recommend it enough. I do not speak any Chineses languages. I only have the highest respect and admiration for China in all things.

Chinese characters are entirel contextual and I thought well if a model can be trained on a bigram  at the very least it will be less fragile. Seems my idea is not so "my" or far our there as there is a project that is similiar called CALLIREADER which came out with some great work recently.


 ##  (dǎ)
  打 (dǎ) (shǒu, hand radical) + 丁 (dīng, a nail or a man). It's one of the most semantically promiscuous characters in Chinese. The radical 扌is the semantic clue—it marks the character as hand-related action—while 丁 provides the phonetic clue (dīng → dǎ).  When the model sees 打 in different bigram contexts, it's not just seeing a random neighbor. It's seeing the character deployed in radically different semantic domains:
  打开 (dǎkāi): to open — physical action
  打电话 (dǎ diànhuà): to make a phone call — abstract, technological
  打扰 (dǎrǎo): to disturb — social/interpersonal
  打架 (dǎjià): to fight — physical conflict
  打针 (dǎzhēn): to give an injection — medical
Same visual form, same hand-radical, but the contextual meaning shifts entirely depending on the neighbor.the model can't just memorize "打 = this shape." It has to learn that 打 is a functional unit whose role is defined by what surrounds it. While a small quantitavly small dataset, the complexity is far greater than the n conveys. The distractor selection in Phase 3 focuses on characters sharing 扌like 打 (to hit), 扔 (to throw), 扛 (to carry on shoulder) make good distractors because they share the hand radical but have different right-hand components. The model has to learn that the right side matters, not just "hand radical = 打."
  
## Dataset sources and access dates

https://huggingface.co/search/full-text?q=CASIA_data%5CHWDB1.1trn_gnt accessed from hugging face 27 May 2026 14:35
https://huggingface.co/search/full-text?q=CASIA_data%5CHWDB1.1tst_gnt accessed from hugging face 27 May 2026  17:25

PyCasia patch


##  manual crop audit results
 (135 good, 17 bad, 1 dropped)

 ## Known edge cases (angular foot, irregular spacing)
  (angular foot, irregular spacing)
  
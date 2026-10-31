"""
Comprehensive multi-domain emotion recognition training and benchmark dataset.
Curated and maintained by Sumit Kumar (https://github.com/Sumit2003-dev)

Covers 8 granular emotional categories:
- joy, sadness, anger, fear, surprise, love, disgust, neutral

Includes realistic multi-domain text: conversational, workplace, social media,
customer sentiment, introspective writing, and complex syntactic patterns (e.g. negations).
"""

import pandas as pd
from typing import List, Tuple

# Curated benchmark dataset samples across 8 emotions
RAW_EMOTION_DATA: List[Tuple[str, str]] = [
    # ==================== JOY ====================
    ("I am so thrilled and ecstatic about winning first place in the competition!", "joy"),
    ("We finally launched our product today and the feedback has been overwhelmingly positive!", "joy"),
    ("Having dinner with my family and laughing until our stomachs hurt was pure bliss.", "joy"),
    ("Congratulations to everyone who worked tirelessly on this project, we did it!", "joy"),
    ("Woke up feeling refreshed and full of vibrant energy for the wonderful day ahead.", "joy"),
    ("I got accepted into my dream university! This is the happiest moment of my life!", "joy"),
    ("The sunrise over the mountains this morning was breathtaking and filled me with gratitude.", "joy"),
    ("Nothing beats the pure delight of eating warm homemade cookies on a cozy afternoon.", "joy"),
    ("Celebrating my best friend's wedding was such a heartwarming and euphoric experience.", "joy"),
    ("I aced my hardest exam with flying colors! All that hard studying finally paid off!", "joy"),
    ("Our team just closed our biggest deal of the quarter, fantastic job everyone!", "joy"),
    ("Adopted a puppy today and she is the sweetest, most playful angel in the world!", "joy"),
    ("Feeling blessed and thankful for all the genuine friends in my corner.", "joy"),
    ("The concert last night was unbelievable, the entire stadium was singing and dancing!", "joy"),
    ("Finally cleared all my student loans today, I feel so liberated and happy!", "joy"),
    ("The vacation in Italy was magical, full of sunshine, gelato, and joyous memories.", "joy"),
    ("Such wonderful news to hear that you are recovering so quickly and feeling great!", "joy"),
    ("Seeing my daughter take her very first steps brought tears of joy to my eyes.", "joy"),
    ("It feels incredible when your hard work and dedication are recognized and appreciated.", "joy"),
    ("Everything is falling into place smoothly and I couldn't be happier right now.", "joy"),
    ("Just received a promotion and a salary raise at work! Life is truly good!", "joy"),
    ("Baking pies with grandma always brings back the fondest, happiest childhood memories.", "joy"),
    ("I'm beaming with pride seeing how much our students have grown this semester.", "joy"),
    ("A peaceful sunny day in the park with a great book and great music, total happiness.", "joy"),
    ("We celebrated our anniversary with a romantic dinner and fireworks, what a joyous night!", "joy"),
    ("The crowd erupted into cheers and celebration when the final goal was scored!", "joy"),
    ("Winning this award is an honor that brings immense happiness to our whole team.", "joy"),
    ("Waking up to birds chirping and blue skies puts an instant smile on my face.", "joy"),
    ("It is so satisfying to finish a complex project and see it work seamlessly.", "joy"),
    ("Laughed so hard with my old friends that we could barely breathe, what an awesome evening!", "joy"),

    # ==================== SADNESS ====================
    ("I feel so lonely, empty, and isolated inside, like nobody truly understands me.", "sadness"),
    ("It breaks my heart to say goodbye to the place I called home for so many years.", "sadness"),
    ("My beloved dog passed away last night and the house feels unbearably silent.", "sadness"),
    ("I worked so hard for months only to be rejected at the final interview stage.", "sadness"),
    ("Losing a close friend to an avoidable tragedy has left me utterly devastated.", "sadness"),
    ("Sometimes the weight of grief feels so heavy that it is hard to get out of bed.", "sadness"),
    ("I feel like an utter failure who let down everyone who ever believed in me.", "sadness"),
    ("Watching them drift away from me slowly without being able to stop it is agonizing.", "sadness"),
    ("Crying quietly in my room so no one can hear how broken I feel right now.", "sadness"),
    ("The diagnosis was not what we hoped for, and our whole family is struggling with sorrow.", "sadness"),
    ("I miss the days when things were simple and we didn't carry this deep sorrow.", "sadness"),
    ("Everything reminds me of you, and it hurts deeply every single day.", "sadness"),
    ("No matter how much effort I put in, things always seem to crumble and fall apart.", "sadness"),
    ("It is exhausting pretending to be okay when on the inside you are falling apart.", "sadness"),
    ("I feel abandoned and forgotten by the people I cared about most.", "sadness"),
    ("The funeral service today was one of the most painful experiences of my life.", "sadness"),
    ("Another rainy day sitting alone staring out the window with tears in my eyes.", "sadness"),
    ("I am so disappointed that our dreams didn't work out after all the sacrifices we made.", "sadness"),
    ("Seeing someone you love suffer in pain without being able to help is true heartbreak.", "sadness"),
    ("Deep down I feel an overwhelming gloom that just won't fade away.", "sadness"),
    ("Regret and sadness are clouding my mind today, wishing I could turn back time.", "sadness"),
    ("My heart sank when I opened that envelope and read the termination letter.", "sadness"),
    ("I'm struggling to find any hope or motivation amidst all this loss and despair.", "sadness"),
    ("They tore down our childhood park, another piece of my past erased forever.", "sadness"),
    ("Feeling totally defeated and helpless in the face of this persistent hardship.", "sadness"),
    ("It's a bitter, melancholy feeling watching everyone move on while you're stuck in grief.", "sadness"),
    ("I miss my grandparents so terribly, the world is much colder without them.", "sadness"),
    ("Felt completely invisible and excluded at the gathering today, made me weep.", "sadness"),
    ("The plans fell through and all our hopes were completely dashed in an instant.", "sadness"),
    ("It hurts to realize that you meant so little to someone who was your whole world.", "sadness"),

    # ==================== ANGER ====================
    ("I am absolutely furious that they lied straight to my face after I trusted them!", "anger"),
    ("This customer support is utter garbage and has wasted four hours of my valuable time!", "anger"),
    ("Stop interrupting me every time I try to speak, it is infuriating and disrespectful!", "anger"),
    ("They stole my idea and presented it to the executive board as their own work, I'm livid!", "anger"),
    ("How dare you blame me for a mistake that was entirely caused by your own negligence!", "anger"),
    ("I am sick and tired of these hypocrites preaching morality while exploiting everyone!", "anger"),
    ("The airline canceled my flight without explanation and refused any refund, unbelievable rage!", "anger"),
    ("It makes my blood boil when people mistreat defenseless animals or vulnerable people.", "anger"),
    ("I have had it with your toxic attitude and constant disrespect in this workplace!", "anger"),
    ("They broke their promise for the third time this week, I cannot tolerate this anymore!", "anger"),
    ("This whole system is rigged and completely corrupt, it makes me so angry I could scream!", "anger"),
    ("Stop gaslighting me and pretending you didn't do anything wrong, take accountability!", "anger"),
    ("I spent weeks preparing this report and you casually deleted it without asking? Are you serious?!", "anger"),
    ("The driver nearly ran over a pedestrian because they were texting, absolute reckless idiocy!", "anger"),
    ("I demand an immediate refund and an apology for this unacceptable treatment!", "anger"),
    ("Don't you ever raise your voice at me or speak to me in that condescending tone again!", "anger"),
    ("It infuriates me that the guilty party walked away with zero consequences while we suffered.", "anger"),
    ("I am boiling with anger after reading their insultingly arrogant email response.", "anger"),
    ("You deliberately sabotaged my presentation to make yourself look better, you coward!", "anger"),
    ("This landlord is trying to unlawfully withhold our deposit, we are taking legal action!", "anger"),
    ("I am so mad right now that my hands are literally shaking with indignation!", "anger"),
    ("Their blatant favoritism and unfair treatment is driving everyone in the department crazy.", "anger"),
    ("I told you repeatedly not to share my private information, you completely violated my trust!", "anger"),
    ("What an outrageous scam, taking money from people and delivering broken junk!", "anger"),
    ("I am fed up with the incompetence and excuses, get it fixed immediately!", "anger"),
    ("Cut in line right in front of elderly folks without even blinking, pure insolence!", "anger"),
    ("I will not sit quietly while you slander my reputation with baseless lies!", "anger"),
    ("Furious that the contractor took half the payment and abandoned the job halfway through.", "anger"),
    ("Your rude and hostile remarks during the meeting were completely unprofessional!", "anger"),
    ("I hate how selfish and inconsiderate some people can be in public spaces!", "anger"),

    # ==================== FEAR ====================
    ("I am terrified that something awful might happen to my family during the storm.", "fear"),
    ("Every little creak in the house tonight is making my heart race with sheer panic.", "fear"),
    ("I have this dreadful feeling that we are heading straight into a catastrophic disaster.", "fear"),
    ("Waiting for the biopsy results is filling me with paralyzing anxiety and dread.", "fear"),
    ("I froze in terror when I saw the shadow moving outside my bedroom window.", "fear"),
    ("The turbulent flight dropped suddenly and passengers started screaming in fear.", "fear"),
    ("I'm frightened about losing my job and being unable to pay rent for my children.", "fear"),
    ("My chest feels tight and I'm trembling with overwhelming panic and dread.", "fear"),
    ("Walking alone through that dark, deserted alley late at night was terrifying.", "fear"),
    ("I have a phobia of heights and looking down from the cliff made me dizzy with fright.", "fear"),
    ("The ominous warning sirens began blaring across the entire city.", "fear"),
    ("I am scared to speak up because of what the retaliatory consequences might be.", "fear"),
    ("A wave of chilling dread washed over me when I couldn't reach my son on the phone.", "fear"),
    ("The doctors said there are serious complications and I am genuinely frightened.", "fear"),
    ("I heard footsteps following closely behind me in the fog and panicked.", "fear"),
    ("Worried sick about the upcoming surgery, I can't sleep or keep food down.", "fear"),
    ("The financial market collapse is terrifying for anyone nearing retirement.", "fear"),
    ("I am anxious that I will freeze up and humiliate myself during the live speech.", "fear"),
    ("The car spun out of control on the black ice, I thought we were going to die.", "fear"),
    ("An eerie silence settled over the woods, and I felt watched by something sinister.", "fear"),
    ("Fearing the worst as the wildfire approaches our neighborhood boundaries.", "fear"),
    ("My hands were trembling and cold as I opened the dreaded audit notification.", "fear"),
    ("I feel unsafe and hypervigilant walking around after the recent neighborhood break-ins.", "fear"),
    ("Nightmares of falling and drowning have been haunting my sleep all week.", "fear"),
    ("Panic attacks are so frightening because you genuinely feel like you cannot breathe.", "fear"),
    ("I dread the thought of having to face my abuser in court next Tuesday.", "fear"),
    ("The sudden earthquake shook the building violently, spreading chaos and fear.", "fear"),
    ("Trembling with fear as the aggressive stray dogs surrounded my bicycle.", "fear"),
    ("I have severe anxiety about what the future holds in this unpredictable crisis.", "fear"),
    ("The look of sheer terror on the child's face when the fireworks exploded nearby.", "fear"),

    # ==================== SURPRISE ====================
    ("Oh wow, I honestly did not expect to see all my friends here for a surprise party!", "surprise"),
    ("What a shock! I never imagined our small startup would get acquired by Google!", "surprise"),
    ("I was completely astonished by the unexpected plot twist in the movie finale!", "surprise"),
    ("Are you serious?! That is the most astonishing and unexpected revelation ever!", "surprise"),
    ("I opened the mailbox and was stunned to find a handwritten letter from 20 years ago!", "surprise"),
    ("No way! Did we really just win the raffle lottery out of ten thousand tickets?!", "surprise"),
    ("I was taken aback by how drastically different the city looked after just five years.", "surprise"),
    ("Holy cow, look at the size of that meteor streaking across the night sky!", "surprise"),
    ("I gasped when the magician made the entire grand piano vanish in thin air!", "surprise"),
    ("Who would have thought that quiet introverted intern was actually a secret chess grandmaster?!", "surprise"),
    ("My jaw literally dropped when I saw the extravagant birthday gift they prepared for me.", "surprise"),
    ("I was bewildered and stunned when my long-lost cousin tapped me on the shoulder at the airport.", "surprise"),
    ("What an unexpected turn of events, nobody saw this announcement coming!", "surprise"),
    ("It caught me completely off guard when the CEO suddenly stepped down this morning.", "surprise"),
    ("I cannot believe my eyes, the northern lights are actually visible from our backyard!", "surprise"),
    ("Wait, really?! You solved in ten minutes what took our team three weeks to debug?!", "surprise"),
    ("I was speechless when they unveiled the breathtaking portrait they painted of me.", "surprise"),
    ("Such an unforeseen discovery in the archaeological dig, this rewrites our history!", "surprise"),
    ("I jumped out of my seat when the phone rang loudly in the dead silence of the night.", "surprise"),
    ("An astonishingly sudden change in weather, from scorching sun to heavy snow in hours!", "surprise"),
    ("I had no clue you were planning this trip, what a wonderful and shocking surprise!", "surprise"),
    ("Well that was unexpected! I thought for sure our team was out of the tournament!", "surprise"),
    ("I rubbed my eyes in disbelief when I saw the bank balance after the inheritance settled.", "surprise"),
    ("Unbelievable, a double rainbow arching right across the waterfall canyon!", "surprise"),
    ("I was startled when the cat leaped out from inside the cardboard box!", "surprise"),
    ("Out of nowhere, an old childhood friend messaged me after fifteen years of silence!", "surprise"),
    ("What a bizarre coincidence that we booked the exact same seat row on the same flight!", "surprise"),
    ("I was amazed by the sheer scale and architectural wonder of the ancient cathedral.", "surprise"),
    ("Never in a million years did I imagine finding my grandmother's lost wedding ring in the garden!", "surprise"),
    ("That sudden twist caught the entire audience completely by surprise!", "surprise"),

    # ==================== LOVE ====================
    ("I love you more than words could ever describe, you are my soulmate forever.", "love"),
    ("Holding my newborn baby against my chest filled my soul with boundless tenderness.", "love"),
    ("You mean the world to me, and I am so grateful to share this life journey with you.", "love"),
    ("The way my partner cares for me when I'm sick melts my heart every single time.", "love"),
    ("Unconditional love is the greatest gift a parent can give to their child.", "love"),
    ("I cherish every single moment we spend together, laughing and dreaming of our future.", "love"),
    ("My heart swells with adoration whenever I look into your kind and gentle eyes.", "love"),
    ("Thank you for being my rock, my confidant, and the love of my life through thick and thin.", "love"),
    ("Cuddling under a warm blanket with the one you love is the purest form of comfort.", "love"),
    ("I am deeply and madly in love with you, and that will never change as long as I live.", "love"),
    ("Our grandparents have been married for sixty years and their devotion to each other is inspiring.", "love"),
    ("Every morning waking up next to you feels like a blessing I never want to take for granted.", "love"),
    ("The bond between twin sisters is filled with deep affection, empathy, and unbreakable love.", "love"),
    ("I adore your warm smile, your sweet laugh, and the immense kindness in your heart.", "love"),
    ("Nothing is stronger than the fierce maternal love a mother holds for her children.", "love"),
    ("You brought romance, compassion, and true affection back into my world.", "love"),
    ("I wrote a love poem expressing how deeply you have touched my mind and spirit.", "love"),
    ("Holding hands while walking along the beach at dusk felt so intimate and romantic.", "love"),
    ("I will always protect, support, and love you no matter what hurdles come our way.", "love"),
    ("Your compassionate embrace made all my fears and worries melt away into safety.", "love"),
    ("I'm so fond of my little rescue kitten, she sleeps right on my chest purring sweetly.", "love"),
    ("True friendship grounded in mutual love and loyalty is rare and priceless.", "love"),
    ("The handwritten vows they exchanged at the altar moved everyone to tears of affection.", "love"),
    ("You are the sweetest, most generous partner anyone could ever wish for.", "love"),
    ("Sending you all my love, warmest hugs, and affectionate kisses across the miles.", "love"),
    ("I fall in love with you a little bit more every single day we are together.", "love"),
    ("My family's constant encouragement and warmth has been my greatest pillar of strength.", "love"),
    ("Her gentle touch and tender voice brought peace and comfort to my troubled heart.", "love"),
    ("Forever thankful for the unending love and guidance my parents gave me.", "love"),
    ("You have my whole heart, today, tomorrow, and for all eternity.", "love"),

    # ==================== DISGUST ====================
    ("The smell of rotting garbage and moldy food in the alleyway was utterly nauseating.", "disgust"),
    ("It sickens me how corrupt politicians line their pockets while ordinary people suffer.", "disgust"),
    ("Found a clump of dirty hair inside my soup at the restaurant, absolutely repulsive!", "disgust"),
    ("The public restroom was filthy, covered in grime and reeked of sewage, revolting!", "disgust"),
    ("I am disgusted by the cruel behavior and mockery displayed by those schoolyard bullies.", "disgust"),
    ("The sour milk was chunky, curdled, and smelled foul, I almost vomited immediately.", "disgust"),
    ("Seeing people spit chewing tobacco on the clean sidewalk fills me with disgust.", "disgust"),
    ("Their slimy, dishonest excuses make me sick to my stomach.", "disgust"),
    ("The basement was infested with roaches, maggots, and decaying rodent carcasses, gross!", "disgust"),
    ("I find his crude, sexist, and degrading remarks utterly offensive and abhorrent.", "disgust"),
    ("The meat was greenish, slimy to the touch, and buzzing with flies, totally putrid.", "disgust"),
    ("It is sickening how corporate scammers prey on vulnerable elderly citizens.", "disgust"),
    ("I couldn't stand the sight of the clogged greasy drain overflowing with filth.", "disgust"),
    ("Your deceitful tactics and fake flattery are utterly distasteful and transparent.", "disgust"),
    ("That horror scene with grotesque body mutilation was too vile and nauseating to watch.", "disgust"),
    ("The kitchen hygiene at that street stall was horrifying, rat droppings near the flour!", "disgust"),
    ("I felt repulsed when he tried to touch my arm after making inappropriate advances.", "disgust"),
    ("The stench of stale beer, vomit, and old cigarette smoke in that dingy bar was nauseating.", "disgust"),
    ("I have total contempt and disgust for anyone who abandons their pets on the side of a highway.", "disgust"),
    ("The slimy texture of that rotten oyster made me gag and spit it out instantly.", "disgust"),
    ("It is abhorrent that toxic chemical waste was dumped illegally into the local drinking river.", "disgust"),
    ("His smug, arrogant grin after getting away with cheating is utterly repulsive.", "disgust"),
    ("The damp carpet had black mold growing in the corners and gave off a sickening stench.", "disgust"),
    ("How can anyone stomach watching those disgusting, exploitative reality television shows?", "disgust"),
    ("I recoil in revulsion whenever I remember that awful, infested motel room.", "disgust"),
    ("The food had expired three months ago and had green furry mold spreading across the bread.", "disgust"),
    ("Such vile and hateful slurs have no place in a civilized society.", "disgust"),
    ("The grease trap behind the kitchen hadn't been emptied in years and smelled sickening.", "disgust"),
    ("I felt dirty and disgusted just being in the presence of such shameless corruption.", "disgust"),
    ("A nauseating odor of sulfur and dead fish lingered over the stagnant canal water.", "disgust"),

    # ==================== NEUTRAL ====================
    ("The monthly financial overview will be reviewed at the scheduled 2:00 PM meeting.", "neutral"),
    ("Please remember to submit your timesheets before the end of the day on Friday.", "neutral"),
    ("The train from platform four stops at all local stations except Oakwood and Elm Grove.", "neutral"),
    ("The chemical formula for water is two hydrogen atoms bonded to one oxygen atom.", "neutral"),
    ("The recipe calls for two cups of all-purpose flour and half a teaspoon of baking soda.", "neutral"),
    ("The library operates from 9:00 AM to 8:00 PM on weekdays and closes at 5:00 PM on weekends.", "neutral"),
    ("The distance between the two cities is approximately one hundred and twenty miles by highway.", "neutral"),
    ("Operating system updates will be installed automatically tonight during maintenance hours.", "neutral"),
    ("The document contains forty-five pages divided into six distinct analytical sections.", "neutral"),
    ("According to the census data, the metropolitan population increased by two percent last year.", "neutral"),
    ("The thermostat in the conference room is currently set to seventy-two degrees Fahrenheit.", "neutral"),
    ("Please attach the requested PDF invoice to the email and copy the procurement department.", "neutral"),
    ("The store carries three different sizes of the storage container in clear plastic.", "neutral"),
    ("The server reboot completed successfully in forty-two seconds without error logs.", "neutral"),
    ("The symposium takes place on the third floor of the engineering building in lecture hall B.", "neutral"),
    ("A standard marathon covers an official distance of twenty-six point two miles.", "neutral"),
    ("The battery level on the device is currently sixty-eight percent.", "neutral"),
    ("The committee reviewed the statutory guidelines and documented their procedural recommendations.", "neutral"),
    ("Water boils at one hundred degrees Celsius at sea level atmospheric pressure.", "neutral"),
    ("The contract expires on December thirty-first and requires sixty days written renewal notice.", "neutral"),
    ("Vehicles must display a valid parking permit when parked in designated zone C.", "neutral"),
    ("The flight duration between Chicago and New York is roughly two hours and fifteen minutes.", "neutral"),
    ("The system requires a password containing at least eight alphanumeric characters.", "neutral"),
    ("The table of contents lists the appendices starting on page one hundred and fourteen.", "neutral"),
    ("The office will remain closed on Monday in observance of the statutory public holiday.", "neutral"),
    ("All participants are requested to wear appropriate identification badges at all times.", "neutral"),
    ("The package was delivered to the reception desk and signed for by building security.", "neutral"),
    ("The average annual rainfall in the valley is approximately twenty-four inches.", "neutral"),
    ("The printer in room two hundred and four is connected to the primary local network.", "neutral"),
    ("Please verify the serial number printed on the back panel before proceeding with installation.", "neutral"),
]

# Additional curated augmentations for syntactic variety & edge cases (negations, idioms, slang)
EXTENDED_EMOTION_DATA: List[Tuple[str, str]] = [
    # Joy variations
    ("I couldn't be more pleased with the outstanding results of our research team!", "joy"),
    ("Such a pleasant surprise to spend the sunny afternoon playing frisbee in the park.", "joy"),
    ("Winning the championship title was the crowning achievement of my athletic career!", "joy"),
    ("Our road trip down the Pacific Coast Highway was filled with endless smiles and epic tunes.", "joy"),
    ("I am so delighted that our hard work is finally paying off in such a big way!", "joy"),
    ("The smile on my grandmother's face when we arrived brought pure warmth and joy to my heart.", "joy"),
    ("We danced until midnight, celebrating our graduation with everyone we love!", "joy"),
    ("I am so proud of how resilient and successful you have become!", "joy"),
    ("The taste of fresh sweet strawberries on a summer day is pure bliss.", "joy"),
    ("Every single goal we set for this milestone was achieved ahead of schedule!", "joy"),

    # Sadness variations (including negations)
    ("I am not happy at all with how things turned out, I just feel numb and down.", "sadness"),
    ("No matter what I do, nothing brings me happiness anymore.", "sadness"),
    ("It hurts so bad to realize that the friendship is officially over.", "sadness"),
    ("I tried so hard to stay positive, but the grief overwhelmed me completely.", "sadness"),
    ("Feeling so down and blue today, nothing seems to spark joy anymore.", "sadness"),
    ("I miss hearing his voice in the morning, the silence in the apartment is deafening.", "sadness"),
    ("A heavy sigh escaped my lips as I packed away the old memories in cardboard boxes.", "sadness"),
    ("I'm not doing well emotionally, everything feels completely hopeless and bleak.", "sadness"),
    ("The feeling of utter helplessness when bad news strikes one after another.", "sadness"),
    ("A tear rolled down my cheek as I read through our old messages from happier times.", "sadness"),

    # Anger variations (including negations)
    ("I am not going to put up with this blatant disrespect for another second!", "anger"),
    ("It is never acceptable to treat your colleagues like second-class citizens!", "anger"),
    ("I am so ticked off that they charged my card twice and refuse to issue a reversal!", "anger"),
    ("Stop lying! You knew exactly what you were doing and you did it intentionally!", "anger"),
    ("This reckless driver almost caused a five-car pileup, take their license away!", "anger"),
    ("I am furious that our wages were frozen while the executives gave themselves massive bonuses!", "anger"),
    ("You have no right to barge in here and make baseless accusations against my staff!", "anger"),
    ("I've reached my absolute boiling point with this constant bureaucracy and delays!", "anger"),
    ("How dare you spread malicious rumors about my personal life behind my back?!", "anger"),
    ("I am raging right now! They wiped our production server without taking a single backup!", "anger"),

    # Fear variations (including negations)
    ("I cannot stop worrying that something dreadful is lurking in the dark shadows.", "fear"),
    ("I'm not confident at all, my gut is warning me that we are in serious danger.", "fear"),
    ("The test results are due tomorrow and my anxiety is completely through the roof.", "fear"),
    ("I was trembling uncontrollably when the turbulence shook the plane from side to side.", "fear"),
    ("The thought of losing our home in foreclosure keeps me terrified every waking hour.", "fear"),
    ("I froze dead in my tracks when I heard a low growl behind me in the woods.", "fear"),
    ("Panic set in when my phone died while I was lost in an unfamiliar city at 2 AM.", "fear"),
    ("I dread going into that dark basement alone after hearing strange scratching sounds.", "fear"),
    ("The sudden blackout plunged the subway car into pitch black and people began screaming.", "fear"),
    ("Terrified of failing this board certification after borrowing thousands of dollars for it.", "fear"),

    # Surprise variations
    ("Holy smoke, who would have predicted such an astonishing turnaround in the stock market?!", "surprise"),
    ("I was completely blown away by the unexpected fireworks show over the harbor!", "surprise"),
    ("Wait a minute, is that really who I think it is standing right in front of me?!", "surprise"),
    ("Never in my wild dreams could I have anticipated this incredible plot twist!", "surprise"),
    ("I opened the closet and out jumped my brother shouting Happy Birthday!", "surprise"),
    ("What a stunning discovery, researchers have uncovered an entire lost city under the jungle!", "surprise"),
    ("My heart skipped a beat when the lightning bolt struck the tree right outside my window!", "surprise"),
    ("I was caught totally off guard when they announced my name as the grand prize winner!", "surprise"),
    ("Unbelievable! The underdog team came back from ten points behind in the final minute!", "surprise"),
    ("I stared in disbelief as the total eclipse turned daytime into starry night in seconds!", "surprise"),

    # Love variations
    ("You are the sweetest, kindest human being and I am so blessed to love you.", "love"),
    ("Holding hands with my grandmother and listening to her stories fills my soul with pure affection.", "love"),
    ("I promise to cherish, respect, and adore you every single day of our lives together.", "love"),
    ("The warmth of a parent's hug can heal even the heaviest emotional wounds.", "love"),
    ("I am so deeply attached to you, your happiness means everything to me.", "love"),
    ("Every little love note you leave on the bathroom mirror brightens my entire morning.", "love"),
    ("A mother's tender touch and soothing lullaby brought calm to the weeping infant.", "love"),
    ("True intimacy is being able to share your deepest vulnerabilities with someone you cherish.", "love"),
    ("I love the way your eyes crinkle when you laugh, you make my heart flutter.", "love"),
    ("Nothing in this universe compares to the gentle warmth of true, reciprocal love.", "love"),

    # Disgust variations
    ("The stench of rotten eggs and sewage coming from that drain is making me gag.", "disgust"),
    ("It is sickening to watch corrupt officials bribe inspectors to overlook safety violations.", "disgust"),
    ("Found a moldy dead insect floating in the bottled beverage, utterly nauseating!", "disgust"),
    ("I am repulsed by how greedily they exploited vulnerable homeless people for views.", "disgust"),
    ("The greasy film covering the kitchen counters and stove was totally repulsive.", "disgust"),
    ("His slimy, deceitful excuses make me want to throw up from sheer revulsion.", "disgust"),
    ("The spoiled milk smelled like sour vinegar and curdled into gross yellow chunks.", "disgust"),
    ("I feel sickened to my core by their cowardly and deceitful betrayal.", "disgust"),
    ("The trash cans were overflowing with maggot-infested scraps in the summer heat.", "disgust"),
    ("I find his vulgar, predatory behavior around young interns completely despicable.", "disgust"),

    # Neutral variations
    ("The conference schedule has been updated to reflect the new keynote speaker room.", "neutral"),
    ("The data was collected over a six-month period across three participating institutions.", "neutral"),
    ("Please save your draft in the shared directory before logging out of the terminal.", "neutral"),
    ("The bus leaves the terminal every fifteen minutes during regular weekday transit hours.", "neutral"),
    ("Standard atmospheric pressure is defined as 101.325 kilopascals at sea level.", "neutral"),
    ("The invoice was processed under reference code INV-88291 on Monday morning.", "neutral"),
    ("The user guide contains diagrams illustrating the step-by-step setup procedure.", "neutral"),
    ("According to the weather bulletin, winds are expected to blow from the northwest at 10 mph.", "neutral"),
    ("The building was constructed in 1984 and renovated during the summer of 2012.", "neutral"),
    ("The application programming interface accepts HTTP POST requests with a JSON payload.", "neutral"),

    # Additional Conversational & Expressive Samples
    # Joy
    ("I'm walking on sunshine and nothing can ruin my mood today!", "joy"),
    ("We won! Unbelievable performance, I am so ecstatic!", "joy"),
    ("A peaceful weekend retreat surrounded by nature brought so much joy and calm.", "joy"),
    ("My heart is bursting with happiness after hearing your wonderful news!", "joy"),
    ("We pulled off the grand surprise flawlessly and everyone was celebrating!", "joy"),
    ("Enjoying hot chocolate by the fire on a snowy evening is total bliss.", "joy"),
    ("Received an offer for my dream position with generous equity, I'm overjoyed!", "joy"),
    ("Such delightful company and tasty food, couldn't ask for a better evening.", "joy"),
    ("Seeing all the children laughing and playing happily made my entire week.", "joy"),
    ("We finished the marathon! My legs ache but my spirit is soaring with joy!", "joy"),

    # Sadness
    ("I feel completely drained, heartbroken, and hopeless right now.", "sadness"),
    ("A deep sorrow settled over me as I watched the empty house one last time.", "sadness"),
    ("I can't stop the tears from flowing, I miss them so terribly much.", "sadness"),
    ("Everything I cared about seems to be slipping away from my hands.", "sadness"),
    ("A devastating loss that has left an irreplaceable void in my heart.", "sadness"),
    ("I feel utterly alone in this big world, surrounded by quiet misery.", "sadness"),
    ("Disheartened and exhausted from battling these endless painful setbacks.", "sadness"),
    ("It broke my spirit to see all our sacrifices amount to nothing in the end.", "sadness"),
    ("A bleak and gloomy day reflecting the quiet sadness inside my chest.", "sadness"),
    ("I'm grieving deeply and trying to take it one difficult minute at a time.", "sadness"),

    # Anger
    ("I am absolutely sick of your arrogant excuses, shut up and listen for once!", "anger"),
    ("They cheated us out of our hard-earned savings, pure criminal fraud!", "anger"),
    ("I am so outraged by this blatant injustice, heads need to roll immediately!", "anger"),
    ("Don't you dare talk down to me like I'm an idiot, check your attitude!", "anger"),
    ("I'm furious! They completely ruined my car and are refusing to pay for damages!", "anger"),
    ("The audacity to steal credit for someone else's labor is utterly infuriating!", "anger"),
    ("I'm at the end of my patience with this dysfunctional, corrupt organization!", "anger"),
    ("They broke the terms of the agreement and laughed in our faces, I am livid!", "anger"),
    ("Stop lying and manipulating the facts to cover up your gross incompetence!", "anger"),
    ("I slammed the phone down in sheer fury after being transferred for the fifth time!", "anger"),

    # Fear
    ("A cold chill of terror ran down my spine when the door handle slowly turned.", "fear"),
    ("I'm so frightened of what the medical scan might reveal tomorrow morning.", "fear"),
    ("My heart was pounding violently against my ribs as the sirens wailed.", "fear"),
    ("I'm having a severe panic attack, my hands are shaking and I can't catch breath.", "fear"),
    ("The thought of confronting them fills me with overwhelming dread and anxiety.", "fear"),
    ("We were stranded in the blizzard with no cell signal, terrified of freezing.", "fear"),
    ("I have an intense dread that our worst fears are about to come true.", "fear"),
    ("Walking through the pitch dark hallway, every shadow felt menacing and scary.", "fear"),
    ("Petrified that we won't be able to protect our family if the dam breaks.", "fear"),
    ("A sickening wave of fear gripped me when the plane dropped hundreds of feet.", "fear"),

    # Surprise
    ("Oh my goodness, I am totally dumbfounded by this shocking announcement!", "surprise"),
    ("Whoa! Did you see that incredible magic trick? How is that even possible?!", "surprise"),
    ("I was completely taken aback when they named me employee of the year!", "surprise"),
    ("Wait, what?! You two got engaged yesterday?! That is such an amazing shock!", "surprise"),
    ("I gasped out loud when the secret trapdoor swung open beneath our feet!", "surprise"),
    ("Never anticipated such an astonishing turn of events in the final seconds!", "surprise"),
    ("My eyes went wide in disbelief when I saw the ancient relic up close!", "surprise"),
    ("I was floored by how phenomenal and unexpected their live performance was!", "surprise"),
    ("What an extraordinary and startling coincidence to run into you in Tokyo!", "surprise"),
    ("I stood there frozen in utter amazement at the magnificent sight before me!", "surprise"),

    # Love
    ("I love you with every fiber of my being, you make life so beautiful.", "love"),
    ("Snuggling with my sleeping toddler fills my whole heart with tender affection.", "love"),
    ("You are my favorite person in the entire world, I adore everything about you.", "love"),
    ("A gentle kiss on the forehead and a loving whisper to soothe away the tears.", "love"),
    ("So deeply grateful for my partner's constant empathy, warmth, and loving care.", "love"),
    ("My family's unconditional love has been the anchor that kept me grounded.", "love"),
    ("Holding hands as we walked under the starlit sky felt like timeless romance.", "love"),
    ("I cherish our late-night conversations and the deep bond we share together.", "love"),
    ("You bring so much tenderness, passion, and genuine love into my life.", "love"),
    ("My sweet pup greeting me at the door with wagging tail is unconditional love.", "love"),

    # Disgust
    ("The stench of rotting roadkill in the heat was unbearable and sickening.", "disgust"),
    ("I was revolted by the greasy, grime-covered utensils at the diner.", "disgust"),
    ("It sickens me to see corporate predators exploit poor families without shame.", "disgust"),
    ("Found a decaying rodent behind the kitchen stove, utterly repulsive!", "disgust"),
    ("His sleazy and predatory comments made my skin crawl with intense disgust.", "disgust"),
    ("The sour, curdled milk had a putrid odor that made me gag instantly.", "disgust"),
    ("I despise dishonest backstabbers who smile in your face while plotting betrayal.", "disgust"),
    ("The overflowing toilet in the public restroom was a vile and sickening sight.", "disgust"),
    ("I felt utter contempt and disgust for their brazen cruelty towards the helpless.", "disgust"),
    ("The slimy mold festering under the damp carpet reeked of rot and mildew.", "disgust"),

    # Neutral
    ("The meeting is scheduled to convene at two o'clock in room four hundred.", "neutral"),
    ("The report includes data gathered between the months of January and June.", "neutral"),
    ("Please verify that your shipping address is up to date in the profile settings.", "neutral"),
    ("The train departs every thirty minutes on weekdays from track number five.", "neutral"),
    ("The document contains five chapters outlining the project specifications.", "neutral"),
    ("The temperature in the laboratory is maintained at twenty-one degrees Celsius.", "neutral"),
    ("Please submit your quarterly expense receipts by the end of business today.", "neutral"),
    ("The software package requires an operating system running a 64-bit kernel.", "neutral"),
    ("The library catalog provides access to over fifty thousand academic journals.", "neutral"),
    ("The committee will deliver its findings in a written statement next Tuesday.", "neutral"),
]

def get_emotion_dataframe() -> pd.DataFrame:
    """
    Returns a comprehensive dataset combining authentic real-world comments
    (from Google Research's GoEmotions benchmark) and curated expressive sentences.
    """
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    real_world_file = data_dir / "real_world_dataset.csv"

    dfs = []
    if real_world_file.exists():
        try:
            df_real = pd.read_csv(real_world_file)
            dfs.append(df_real)
        except Exception as e:
            print(f"Notice: Failed to load real_world_dataset.csv: {e}")

    # Also add curated expressive benchmark data
    all_data = RAW_EMOTION_DATA + EXTENDED_EMOTION_DATA
    df_curated = pd.DataFrame(all_data, columns=["text", "emotion"])
    dfs.append(df_curated)

    combined = pd.concat(dfs, ignore_index=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined["emotion"] = combined["emotion"].astype(str).str.lower().str.strip()
    
    # Filter out empty or too short
    combined = combined[combined["text"].str.len() > 3]

    return combined.drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=42).reset_index(drop=True)

if __name__ == "__main__":
    df = get_emotion_dataframe()
    print(f"Loaded {len(df)} samples across {df['emotion'].nunique()} classes:")
    print(df["emotion"].value_counts())

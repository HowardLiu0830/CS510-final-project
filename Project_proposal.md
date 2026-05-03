#project #proposal

Project Title: Development: Research Trail Builder and Guided Exploration of Scientific Literature

Team Members: Lawrence Wang (lw41), Eric Chen (ericzc2), Haoyang Wang (hw86), Howard Liu (yl140)

Project Coordinator: Eric Chen (ericzc2)

Project Description: A system that transforms academic search results into interactive, graph-based knowledge maps to help researchers navigate complex research papers and identify research gaps.

[Functions and Users] Define clearly what software tool you’re planning on implementing. Clarify whether you plan to develop a standalone software tool, a new Web-based application, a new mobile app,  an extension of an existing toolkit, or an extension of an existing application system. What are the major functions of the envisioned tool? Who are the users of your software tool/system?
We want to implement a web application that transforms how researchers navigate scientific literature. Rather than returning a flat ranked list of search results, the system constructs an interactive, graph-based knowledge map that visualizes how concepts, methods, and claims interconnect across papers. Some core functions are: (1) topic-driven research scoping, where users initialize a query and the system decomposes it into sub-problems; (2) automated literature discovery and collection via real academic search APIs (ex. Semantic Scholar, OpenAlex); (3) intelligent screening and knowledge extraction, surfacing key claims, methods, and relationships from collected papers; and (4) synthesis into a navigable concept graph that users can explore, annotate, and expand. The primary users are graduate students and members in academia and industry who work on research.

[Significance] Why do we need the tool/system that you propose to develop? Does your tool/system address any existing "pain point"? How would our world be different because of your new tool/system? Does it address a societal need? Why is it important to address this need?
There are millions of papers that are published every year in every field, which means that reviewing all this literature can be overwhelming. Existing tools like Google Scholar and Semantic Scholar return ranked lists, but don’t give guidance as to how ideas actually relate to each other, evolve, or conflict. Researchers would spend weeks manually constructing mental maps of a field before they can meaningfully contribute to it, so our project directly addresses this pain point by automating the connective labor of literature review to reduce cognitive load and visualize the conceptual structure of a field rather than just give us the documents. It can also benefit junior researchers and students by lowering the barrier to entering unfamiliar fields, which is beneficial for both learning and scientific progress.

[Approach] How do you plan to build it? If this is a contribution to an existing piece of software, you should try to know the procedure for contributing to that software.  If this is a standalone tool, identify what technologies you plan on leveraging to implement your software. This may be programming languages, supporting libraries, etc. What existing resources can you leverage? What risk or potential barrier do you anticipate and how do you plan to mitigate the risk? 
Python and LangGraph for agent orchestration framework
Web application: React, Flask, or Streamlit etc
Academic database: OpenAlex, Semantic Scholar, Google Scholar, arXiv APIs
LLM API: Claude, OpenAI, Gemini
Risks:
API rate limits on academic search (mitigate by caching and having fallback abstract-only mode)
LLM hallucination in knowledge extraction (mitigated by grounding all claims with source citations and a confidence scoring)

[Evaluation] How will you demonstrate the usefulness of your tool/system and correctness of your implementation? 
LLM as a judge with specific rubric covering relevance, coverage, structural organization, insightfulness.
Human evaluation with domain-expert participants using a structured rubric to assess relevance, coverage, structural organization, and insightfulness in supporting literature understanding.

[Timeline] Provide a rough timeline to show when you expect to finish what. List a couple of milestones if possible (they can be tentative).
Week 1 (4/7): Environment set up, get agent framework to work on basic examples, set up rough web interface and repo
Week 2 (4/14): Finish core functions 1 2
Week 3 (4/21): Finish core functions 3 4
Week 4 (4/28): Integrate all core functions into the full system, refine the interface, and perform internal testing and error analysis. Improve system quality and prepare the evaluation setup.
Week 5 (5/5): Conduct LLM-based and human evaluation, and analyze results
Week 6 (5/12): Finalize project, prepare demo and presentation

[Task division] Use one sentence to describe what each team member is expected to work on (can be tentative).
Lawrence: core function implementation, agent framework and backend pipeline
Eric: core function implementation, searching academic database, paper parsing and extraction, web interface
Haoyang: core function implementation, focus on summary and flowchart generation, including prompt design and structured output refinement
Howard: core function implementation, evaluation setup for both LLM-based and human assessment

Project Presentation Submission Link: https://mediaspace.illinois.edu/.... (To be completed later; after you finish the presentation submission here, add the hashtag "#presentation")

Project Report Submission Link : https://github.com/... (To be completed later; after you finish the report submission here, add the hashtag "#report")

# Embedding Model per Knowledge Graph: Classificazione di Triple ed Estrazione di Conoscenza

Lavoro di tesi di Balice Matteo.

## Contenuto della Repository

1. **data:** contiene i dataset utilizzate per l'addestramento e test.
2. **JavaReasoner**: è il reasoner contenente le ontologie per derivare gli assiomi.
3. **prediction**: contiene gli script utilizzati per il task di triple classification.
4. **explainability**: contiene gli script utilizzati per l'estrazione di conoscenza.
5. **src**: contiene i modelli TransC-OWL e TransM.


## Sommario

Il web semantico è un’estensione del World Wide Web in un ambiente in cui le risorse al suo interno sono associate a informazioni e dati che ne specificano il contesto semantico in un formato adatto al ragionamento
automatico da parte delle macchine.
La struttura che viene utilizzata per rappresentare graficamente il semantic web è il Knowledge Graph, una struttura basata su grafo in
grado di rappresentare la conoscenza: i nodi rappresentano gli individui (entità, oggetti, concetti...) mentre gli archi rappresentano le relazioni che legano i vari individui.
La chiave del successo di questa rappresentazione è da ritrovare nella loro flessibilità, in quanto consentono di collegare risorse tra loro di natura eterogenea. D’altro canto, però, uno dei principali problemi del Semantic Web è l’incompletezza dei dati: a causa anche dell’enorme mole di dati che bisogna gestire nei Knowledge Graph è impossibile disporre a priori di tutta la conoscenza. Una soluzione a questo problema è il link prediction, cioè predire le relazioni mancanti all’interno dei Knowledge Graph (e triple classification che invece predice la verità di date triple). A questo scopo, sono stati proposti molti modelli, utilizzando anche tecniche molto differenti tra di loro. Tuttavia, molti di questi modelli non riescono a fornire una spiegazione del risultato ottenuto
dalla predizione, il che potrebbe rivelarsi un requisito fondamentale in alcuni domini (come in ambito medico). Obiettivo della tesi è quindi quello di risolvere il problema dell’incompletezza dei Knowledge Graph tramite triple classification analizzando e confrontando le performance di diversi modelli e valutarne la loro interpretabilità (estrazione di conoscenza). Nello specifico verranno considerati due modelli orientati alle classi: TransC-OWL, basato su TransC a cui si è iniettato conoscenza di fondo e TransM.

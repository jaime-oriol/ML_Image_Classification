Welcome to the Machine Learning Assignment for this season 2025-2026!

This season will be focused on practical issues about Image Classification.

The assignment will consist of run a Machine Learning model to clasify a dataset with pictures. You'll do it adapting the "Clasifying Outfit with Pytorch" to run an image classification problem chosen by you. 

You can choose any  of the dataset provided by Pytorch in its Torchvision module to hand in the assignment. Torchvision is plenty of datasets you can play with.

IMPORTANT: CIFAR10 and CIFAR100 ARE FORBIDDEN. DON'T USE THEM TO DO THIS ASSIGNMENT. 

ASSIGNMENTS BASED ON CIFAR10 OR CIFAR100 WILL BE REJECTED.

Please visit the following link to have a glance at them:


https://docs.pytorch.org/vision/main/datasets.html

How do I do the assignment?

These are the instructions you'll follow to hand in your assignment.

1. Form a team with your colleagues. Please, no more than four members in each team.

2. Nominate a speaker, he or she will  be in charge of presenting your work at then end of the assignment.

3. Send me a meesage with the team members. Please specify who will be the speaker.

4. From the Datasets provided by the TorchVision module, choose one of it. If you want to choose another dataset from a different source, feel free to do it!

5. Adapt the code studied in the "Clasisfying Outfit with Pytorch" lesson to deal with the chosen dataset.

6. Build your own Machine Learning model to run the classification, more than one Machine Learning model will be rewarded.

7. Train the model.

8. Test the model, present the results of the testing (choose a metric a explain the results in terms of the chosen metric).

9. Load your own picture a check out how the model works.

 

What should I hand in as the evidence of my assignment?

Every single member of the team, I repeat every single member of the team must upload the following files to CANVAS:

1. The code the team has written down to run the model.

2. The presentation (PowerPoint, Canva, ...) the speaker will follow to show your work.

I repeat again, every single member of the team must upload the files mentioned above. I will not assess the work of the student who hasn't upload the files mentioned above.

The code must be written in Python, using Pytorch framework. You can reuse the code we've studied in the "Clasifying Outfit with Pytorch" lesson.

How will you evaluate the assignment?

 After the due date, I will summon every team to evaluate the assignment. Here it is the procedure I'll follow to do it:

1. The speaker will show the presentation explaining the work the team has done. The presentation must contain the following topics:

1.1 A detailed explanation of the dataset you've chosen (number of observations, how the picture is formed, how the dataset represents the labels, train/test split)

1.2 A description of the Machine Learning model you've built.

1.2 A description of the training stage (learning rate, number of epochs, loss function chosen, optimizacion function) and any interesting topic you'll consider it's worth mentioning it.

1.3 A description of the testing results.

1.4 A description of the picture you've chosen. Has the model capture what the picture you've uploaded it is?

2. The speaker will run the whole code the team has uploaded to Canvas. The code must run end-to-end. If the code doesn't run, the team presentation will finish and the team will not pass the assignment.

3. After we've seen runs properly, I will ask threes question to every single member of the team but the speaker. The speaker doesn't need to ask any question. With these questions I want to check out if every single member of the team properly knows the code.

How will you evaluate the assignment?

The code will be evaluated as following:

The code doesn't run. The team doesn't pass the assignment and therefore the course. The must repeat the assignment in June (Exam score will be kept).
2 pts: the code runs properly.
3 pts: quality of the presentation. Clarity and accuracy in the presentation will be positive evaluated.
3 pts: individual questions made to the every single member of the team. The speaker will have by default these 3 pts witout questions. If the team member is not able to answer any of the three questions, these team member will not pass the assignment and he or she will try it again in June.
2 pts: this points that lead to the maximum score will only be got by teams who build their own image dataset.  
Due date is 08th of January 2025.

I'll answer any doubt in the next class (on Friday 21th of November)

Good luck and let's do it!

Your Machine Learning professor.

----

MI DATA Y ENFOQUE

El dataset final se construyó integrando tres fuentes dentro del mismo dominio geológico: minerales, rocas y texturas naturales. Para mantener coherencia visual y equilibrio entre categorías, los minerales (calcite, pyrite y quartz) se mantuvieron como clases independientes debido a su identidad morfológica clara y su volumen elevado; todas las rocas se agruparon en una única clase para evitar la sobre–representación interna; y las tres texturas superficiales (cracked, porous y wrinkled) se unificaron en una sola clase de “superficies texturizadas”, reforzada con data augmentation para igualar su tamaño.

La construcción de este dataset es útil porque permite abordar un problema realista de clasificación de imágenes dentro de un dominio cohesionado pero heterogéneo como es el geológico. Combinar tres secciones distintas,minerales, rocas y superficies texturizadas, introduce una variabilidad controlada: los minerales aportan clases bien definidas y visualmente ricas; las rocas proporcionan un conjunto amplio con alta diversidad interna que obliga al modelo a generalizar; y las texturas, más escasas, permiten evaluar la eficacia de técnicas de data augmentation para compensar la falta de datos. Esta mezcla intencionada de abundancia, diversidad y escasez balanceada mediante aumento sintético reproduce condiciones habituales en proyectos reales, donde los datasets no son uniformes y requieren decisiones de diseño. El resultado es un entorno de entrenamiento sólido, equilibrado y metodológicamente representativo de los retos reales en visión por computador.
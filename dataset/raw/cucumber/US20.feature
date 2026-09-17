Feature: US20 - Soporte técnico accesible para resolver problemas con la tecnología

  Scenario: Solicitud de soporte técnico remoto
    Given que el usuario necesita asistencia, 
    When haga una solicitud de soporte,
    Then un técnico debe contactarlo en un plazo máximo de 24 horas.

    Examples:
    
    | Formulario de Ticket Soporte |
    | Nombre de usuario |
    | Tipo de solicitud de soporte |
    | Problema que tiene el usuario |

  Scenario: Guía paso a paso
    Given que el usuario tiene dudas sobre cómo usar la aplicación, 
    When acceda a la sección de soporte, 
    Then debe encontrar un tutorial fácil de seguir con respuestas a problemas comunes.

    Examples:

    | Gias o tutoriales |
    | Guia 1 |
    | Guia 2 |
    | Guia 3 |

    | Guia 1 |    

    | Imagen de la herramienta | Descripción de la herramienta |

    | Como llegar a la herramienta |
    | Imagenes de como llegar a la herramiena | Ruta de herramienta de manera escrita |

    | Como usar la herramienta |
    | Video tutorial de la herramienta |

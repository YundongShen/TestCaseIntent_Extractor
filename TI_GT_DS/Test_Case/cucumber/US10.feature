Feature: US10 - Control de gastos por cultivo

  Scenario: Ingreso de gastos individuales
    Given que el usuario compró fertilizante,
    When registra “Compra de fertilizante – S/ 150” en el cultivo de papa, 
    Then el gasto se asocia a ese cultivo y se suma al total invertido.

    Examples:

    | Cultivos asociados | Cultivo 1 | Cultivo 2 | ... |
    | Concepto de gasto N°1 | Monto de gasto |
    | Concepto de gasto N°2 | Monto de gasto |
    | Concepto de gasto N°3 | Monto de gasto |

    | Registro de gastos en la base de datos | Actualización ddel total invertido |

  Scenario: Reporte de gastos acumulados
    Given que el usuario selecciona un cultivo, 
    When accede al panel financiero, 
    Then visualiza un resumen de todos los gastos realizados con fechas, conceptos y montos.

    Examples:
    
    | Seleccion de cultivo | Cultivo Seleccionado |
    | Periodo deseado | Fecha inicio ded periodo | Fecha final del periodo |

    | Tabla dada en el Reporte |
    | Concepto de gasto | Monto del consepto |
    | Concepto de gasto N°1 | Monto del consepto N°1 |
    | Concepto de gasto N°2 | Monto del consepto N°2 |
    | Concepto de gasto N°3 | Monto del consepto N°3 |
    | Inversión total | Monto total invertido |

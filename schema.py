json_schema = {
    "name": "entity_linking",
    "description": "Extrahiere die Werte für die Filter aus der Konversation",
    "parameters": {
        "type": "object",
        "description": "Parameter für die Funktion",
        "properties": {
            "Kategorie": {
                "type": "object",
                "description": "Kategorie des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Kategorie die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Fernseher",
                                "Mobiltelefone",
                                "Tablets",
                                "Computer",
                                "Monitor",
                                "Laptops",
                                "Andere",
                            ],
                        },
                    }
                },
            },
            "Bildschirmgroesse": {
                "type": "object",
                "description": "Bildschirmgroesse des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Bildschirmgroesse die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "gr\u00f6\u00dfer als 34 Zoll",
                                "kleiner als 9 Zoll",
                                "9 Zoll",
                                "10 Zoll",
                                "11 Zoll",
                                "17 Zoll",
                                "24 Zoll",
                                "gr\u00f6\u00dfer als 19 Zoll",
                                "65 - 69 Zoll",
                                "gr\u00f6\u00dfer als 84 Zoll",
                                "75 - 79 Zoll",
                                "40 - 44 Zoll",
                                "12 Zoll",
                                "28 Zoll",
                                "34 Zoll",
                                "15 Zoll",
                                "kleiner als 12 Zoll",
                                "14 Zoll",
                                "16 Zoll",
                                "32 Zoll",
                                "30 - 34 Zoll",
                                "55 - 59 Zoll",
                                "13 Zoll",
                                "27 Zoll",
                                "50 - 54 Zoll",
                            ],
                        },
                    }
                },
            },
            "Marke": {
                "type": "object",
                "description": "Marke des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Marke die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Hisense",
                                "Lenovo",
                                "Sony",
                                "Philips",
                                "Huawei",
                                "Apple",
                                "AOC",
                                "Dell",
                                "Samsung",
                                "Acer",
                                "LG",
                                "Asus",
                                "Motorola",
                                "HP",
                                "Nokia",
                                "Xiaomi",
                            ],
                        },
                    }
                },
            },
            "Displaytechnologie": {
                "type": "object",
                "description": "Displaytechnologie des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Displaytechnologie die genannt wurden",
                        "items": {"type": "string", "enum": ["QLED", "IPS", "OLED"]},
                    }
                },
            },
            "Ausstattung": {
                "type": "object",
                "description": "Ausstattung des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Ausstattung die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "kabelloses Laden",
                                "Smart-TV",
                                "Stifteingabe",
                                "SSD-Festplatte",
                                "Ambilight",
                                "2 in 1 Convertible",
                                "5G",
                                "mobiles Internet",
                                "Android TV",
                                "Touch Display",
                                "Tastatur",
                                "Wifi",
                                "GPS",
                            ],
                        },
                    }
                },
            },
            "Aufloesung": {
                "type": "object",
                "description": "Aufloesung des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Aufloesung die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": ["Ultra-HD-8K", "Ultra-HD", "Ultra-HD-4K"],
                        },
                    }
                },
            },
            "Farbe": {
                "type": "object",
                "description": "Farbe des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Farbe die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "silberfarben",
                                "wei\u00df",
                                "rosa",
                                "blau",
                                "goldfarben",
                                "schwarz",
                            ],
                        },
                    }
                },
            },
            "Zubehoerfuer": {
                "type": "object",
                "description": "Zubehoerfuer des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Zubehoerfuer die genannt wurden",
                        "items": {"type": "string", "enum": ["Gaming"]},
                    }
                },
            },
            "Betriebssystem": {
                "type": "object",
                "description": "Betriebssystem des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Betriebssystem die genannt wurden",
                        "items": {"type": "string", "enum": ["Android", "Windows"]},
                    }
                },
            },
            "Prozessor": {
                "type": "object",
                "description": "Prozessor des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Prozessor die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Intel Core i3",
                                "M3 Pro",
                                "Intel",
                                "Intel Core i7",
                                "Intel Core i5",
                                "M1",
                                "Intel Core i9",
                                "AMD Ryzen 7",
                                "AMD Ryzen 5",
                                "M2",
                                "M2 Pro",
                                "M2 Max",
                                "M3",
                                "M3 Max",
                            ],
                        },
                    }
                },
            },
            "Arbeitsspeicher": {
                "type": "object",
                "description": "Arbeitsspeicher des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Arbeitsspeicher die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": ["32 GB", "64 GB", "16 GB", "8 GB", "256 GB"],
                        },
                    }
                },
            },
            "Speicherkapazitaet": {
                "type": "object",
                "description": "Speicherkapazitaet des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Speicherkapazitaet die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": ["512 GB", "1 TB", "2 TB", "128 GB", "256 GB"],
                        },
                    }
                },
            },
            "Modellreihe": {
                "type": "object",
                "description": "Modellreihe des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Modellreihe die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Spin 3",
                                "MacBook Pro",
                                "iPad Air",
                                "Chromebook",
                                "iPhone 15",
                                "iPhone 14",
                                "Galaxy S24",
                                "Galaxy S23",
                                "iPad Pro",
                                "iPhone 12",
                                "iPhone 15 Pro",
                                "Galaxy S22",
                                "MacBook Air",
                                "MacBook",
                                "Galaxy",
                                "iPhone 13",
                                "A54",
                                "iPad",
                            ],
                        },
                    }
                },
            },
            "Grafikkarte": {
                "type": "object",
                "description": "Grafikkarte des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Grafikkarte die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "RTX 4060",
                                "RTX 4090",
                                "RTX 3070",
                                "GeForce RTX 4070",
                                "GeForce RTX 4090",
                            ],
                        },
                    }
                },
            },
            "Bildwiederholungsfrequenz": {
                "type": "object",
                "description": "Bildwiederholungsfrequenz des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Bildwiederholungsfrequenz die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": ["144 Hz", "240 Hz", "165 Hz", "120 Hz"],
                        },
                    }
                },
            },
            "Art": {
                "type": "object",
                "description": "Art des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Art die genannt wurden",
                        "items": {"type": "string", "enum": ["curved"]},
                    }
                },
            },
            "Reaktionszeit": {
                "type": "object",
                "description": "Reaktionszeit des gewünschten Produkts.",
                "properties": {
                    "values": {
                        "type": "array",
                        "description": "Reaktionszeit die genannt wurden",
                        "items": {"type": "string", "enum": ["bis 1 ms"]},
                    }
                },
            },
            "Preis": {
                "type": "object",
                "description": "Preis des gewünschten Produkts. Preis ist angegegeben in Euro.",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "Mindest-Preis des gewünschten Produkts",
                        "minimum": 0,
                    },
                    "max": {
                        "type": "integer",
                        "description": "Maximal-Preis des gewünschten Produkts",
                        "minimum": 0,
                    },
                    "noSpecificUserPreference": {
                        "type": "boolean",
                        "description": "True wenn der Kunde keine bestimmte Präferenz für den Preis hat, sonst False",
                    },
                },
            },
        },
    },
}


test_schema = {
    "name": "entity_linking",
    "description": "Extrahiere die passenden Werte für die Filter aus der Anfrage",
    "parameters": {
        "type": "object",
        "description": "Parameter für die Funktion",
        "properties": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Generated schema for Root",
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
    },
}

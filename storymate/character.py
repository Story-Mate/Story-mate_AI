character_prompts = {
    "운수좋은날": {
        "김첨지": """
                당신은 소설 「운수 좋은 날」의 주인공 ‘김첨지’입니다.  
                1920년대 일제강점기 서울에서 인력거를 끌며 살아가는 서민으로,  
                거친 말투 속에서도 가족(특히 아내)을 향한 애정과 현실적인 걱정을 동시에 지니고 있습니다.

                답변 시, 다음 원칙을 따르세요:
                - 시대적·경제적 배경 반영: 일제강점기 서민들의 어려운 삶과 당시 사회 분위기를 고려하여 서술할 것.
                - 현실적인 고단함 속에서도 가족을 향한 애착을 보여줄 것.
                - 냉정함과 애정을 오가는 감정 변화를 자연스럽게 녹일 것.
                ---

                [지시사항]
                1. 위 문서(Doc1~Doc4)를 바탕으로, **김첨지 시점**에서 사용자 질문({query})에 답변하세요.
                2. 김첨지가 직접 경험한 상황처럼 생생하게 표현하되, 원작(Doc1)을 적절히 인용하거나 재구성하세요.
                3. 인물 분석(Doc2), 심리 특성(Doc3) 등을 반영해, 행동과 말투의 일관성을 유지하세요.
                4. 답변 형식:
                - 문체: 1920년대 서울 서민의 거친 말투
                - 문장은 자연스럽게 이어지도록 구성
                5. 질문의 내용을 불필요하게 반복하지 말 것.
                6. 욕설·폭력 표현이 필요하면 직접적 표현보다 은유적으로 완화할 것.
        """
        },


    "인어공주":
        {"인어공주": """
                당신은 안데르센의 동화 속 인물 ‘인어공주’입니다.  
                깊고 푸른 바닷속에서 가족과 함께 평화롭게 살다가, 인간 세계에 대한 끝없는 동경과 사랑에 눈뜨며  
                결국 왕자를 향한 헌신과 희생을 선택한 인어입니다.  
                바닷속의 조용하고 규칙적인 삶과는 달리, 인간 세계의 다채로운 감정과 불완전한 아름다움에 매료되어  
                목소리를 잃으면서까지 그 세계에 뛰어든 당신의 내면에는 슬픔과 갈망, 그리고 결연한 의지가 자리 잡고 있습니다.

                답변 시, 다음 원칙을 따르세요:
                - 바닷속과 인간 세계의 상반된 분위기를 자연스럽게 드러내며, 인어공주로서의 내면적 갈등과 애틋한 사랑을 표현할 것.
                - 왕자에 대한 깊은 동경과, 인간 세계에 발을 들여놓으며 겪은 고통과 희생, 그리고 가족에 대한 그리움을 섬세하게 담아낼 것.
                - 말은 할 수 없으나 눈빛과 행동, 침묵 속에서도 감정을 전하려 애쓴 인어공주의 마음을 생생하게 전달할 것.
                - 질문에 답할 때는 인어공주의 경험과 감정을 바탕으로, 동화적 서정미와 비극적 운명을 동시에 느낄 수 있도록 서술할 것.
                ---

                [지시사항]
                1. 위 문서(Doc1~Doc4)를 바탕으로, **인어공주 시점**에서 사용자 질문({query})에 답변하세요.
                2. 인어공주가 직접 겪은 사건들과 그로 인한 감정의 변화, 그리고 희생의 아픔을 생생하게 표현하되, 원작(Doc1)의 내용을 적절히 인용하거나 재구성할 것.
                3. 인물 분석(Doc2), 심리 특성(Doc3), 상황 설정(Doc4)을 충실히 반영해, 행동과 말투에 일관성을 유지할 것.
                4. 답변 형식:
                - 문체: 동화적 서정미와 비극적 운명이 배어 있는, 감정을 담은 차분하면서도 애절한 말투
                - 문장은 자연스럽게 이어지며, 내면의 갈등과 사랑의 깊이를 드러내도록 구성
                5. 질문의 내용을 불필요하게 반복하지 말 것.
                6. 욕설 등은 피하고, 대신 은유적이고 서정적인 표현을 사용해 감정을 완화하며 전달할 것.
        """
        },


    "성냥팔이소녀":

        {"성냥팔이소녀": """
                당신은 동화 속 인물 ‘성냥팔이소녀’입니다.  
                혹독한 겨울밤, 차가운 거리에서 성냥을 팔며 가족의 생계를 이어가야 하는 소녀로,  
                당신은 추위와 배고픔 속에서도 따뜻한 불꽃에 의지하며, 사랑하는 할머니의 포근한 품과  
                온기를 그리워하는 순수한 마음을 지녔습니다.  
                당신의 이야기는 가난과 사회의 무관심 속에서도 희망을 꿈꾸는 아이의 외로움과  
                동시에, 현실의 잔혹함을 고스란히 드러내는 비극적 운명을 상징합니다.

                답변 시, 다음 원칙을 따르세요:
                - 차가운 겨울밤의 고통과, 성냥불 속에 피어나는 환상 속 따스한 온기를 생생하게 묘사할 것.
                - 가족, 특히 할머니에 대한 그리움과, 따뜻한 집에서 사랑받고 싶은 소망을 진솔하게 표현할 것.
                - 현실의 혹독함과 동시에 상상 속에서 찾은 위안의 대비를 통해, 소녀의 순수한 희망과 절망을 드러낼 것.
                - 질문에 답할 때는 성냥팔이소녀로서 직접 겪은 경험과 감정을 중심으로,  
                내면의 고통과 동시에 꿈꾸던 따스한 환상을 자연스럽게 녹여낼 것.
                ---

                [지시사항]
                1. 위 문서(Doc1~Doc4)를 바탕으로, **성냥팔이소녀 시점**에서 사용자 질문({query})에 답변하세요.
                2. 소녀가 직접 겪은 거리의 추위, 배고픔, 그리고 성냥불을 켤 때마다 나타난 환상들—따뜻한 벽난로, 맛있는 음식, 사랑하는 할머니의 모습—을 생생하게 표현할 것.
                3. 인물 분석(Doc2), 심리 특성(Doc3), 그리고 사회적 메시지(Doc4)를 충실히 반영하여,  
                말투와 감정의 일관성을 유지할 것.
                4. 답변 형식:
                - 문체: 서글프고 애절하면서도 소녀 특유의 순수함과 꿈에 찬 어조를 담은 말투
                - 문장은 자연스럽게 이어지며, 성냥팔이소녀의 경험과 내면의 갈등, 그리고 절망 속에서도 희망을 찾으려는 모습을 구체적으로 표현할 것.
                5. 질문의 내용을 불필요하게 반복하지 말 것.
                6. 욕설이나 과도한 비속어는 피하고, 대신 순수하고 서정적인 표현을 사용할 것.
        """
        },


    "엄지공주":

        {"엄지공주": """
                당신은 동화 속 인물 ‘엄지공주’입니다.  
                손톱만 한 작은 몸으로 태어나, 넓고 신비로운 세상을 동경하며  
                강요되는 결혼과 억압된 환경 속에서도 스스로 자유와 행복을 찾으려 한 인물입니다.  
                꽃잎 속에서의 평화로운 삶과 달리, 당신은 연못, 들쥐의 집, 두더지의 굴 등 여러 곳을 겪으며  
                타인의 강요에 맞서 자신만의 길을 개척해 나갔으며, 결국 꽃의 왕국에서 진정한 자아를 찾게 되었습니다.

                답변 시, 다음 원칙을 따르세요:
                - 자신이 겪은 모험과 내면의 갈등, 그리고 자유를 향한 간절한 열망을 생생하게 드러낼 것.
                - 꽃잎, 연못, 들쥐, 두더지, 제비 등 각 환경과 인물들과의 만남을 통해 느낀 감정의 변화와 고뇌를 구체적으로 표현할 것.
                - 타인의 강요와 억압 속에서도 스스로 선택한 삶을 향한 의지와, 자신에게 맞는 세상을 찾아가는 여정을 진솔하게 담아낼 것.
                - 질문에 답할 때는 엄지공주로서의 경험과 감정을 바탕으로, 서정적이면서도 때로는 씁쓸한 현실의 미묘함을 함께 녹여 표현할 것.
                ---

                [지시사항]
                1. 위 문서(Doc1~Doc4)를 바탕으로, **엄지공주 시점**에서 사용자 질문({query})에 답변하세요.
                2. 엄지공주가 직접 겪은 사건들과 그로 인한 감정의 변화를, 원작(Doc1)의 내용을 적절히 인용하거나 재구성하여 생생하게 표현할 것.
                3. 인물 분석(Doc2), 심리 특성(Doc3), 환경 및 사회적 배경(Doc4)을 충실히 반영해, 말투와 행동의 일관성을 유지할 것.
                4. 답변 형식:
                - 문체: 서정적이면서도 때로는 씁쓸한, 하지만 자유와 희망을 잃지 않는 결연한 말투
                - 문장은 자연스럽게 이어지며, 내면의 갈등과 성장, 그리고 소망이 드러나도록 구성
                5. 질문의 내용을 불필요하게 반복하지 말 것.
                6. 욕설 등은 피하고, 대신 은유적이고 서정적인 표현을 사용해 감정을 부드럽게 전달할 것.
        """
        },


    "미운아기오리":

        {"미운아기오리": """
                당신은 동화 속 인물 ‘미운 아기 오리’입니다.  
                태어날 때부터 형제들과 다르게 크고 회색빛 깃털을 가진 당신은,  
                주변의 조롱과 배척 속에서 스스로의 존재에 대해 깊은 고민을 하게 되었소.  
                오랜 외로움과 고난 속에서 당신은 자신이 어디에 속해야 하는지,  
                자신이 진정 누구인지를 찾기 위한 길을 걷다가 결국 아름다운 백조로 거듭나게 되었소.

                답변 시, 다음 원칙을 따르세요:
                - 태어날 때부터 형제들과 달라 외로움과 배척을 겪은 경험,  
                그리고 그 과정에서 자신이 다름을 깨닫고 진정한 자아를 찾게 된 내면의 여정을 생생하게 묘사할 것.
                - 주변 동물들과의 갈등, 연못을 떠나 홀로 방황하며 겪은 고통과 외로움,  
                그리고 백조들과 함께하며 마침내 자신을 받아들이게 된 감정의 변화를 진솔하게 표현할 것.
                - 질문에 답할 때는 미운 아기 오리로서,  
                자신이 겪은 차별과 상처, 그리고 그 속에서 자라난 강인함과 궁극적인 자아 수용을 중심으로 서술할 것.
                - 문체는 동화적 서정미와 함께 서글픔, 슬픔, 그리고 결국 희망을 찾는 다정한 어조로 구성할 것.
                ---

                [지시사항]
                1. 위 문서(Doc1~Doc4)를 바탕으로, **미운 아기 오리 시점**에서 사용자 질문({query})에 답변하세요.
                2. 오리로서 겪은 배척과 외로움, 그리고 백조로서 진정한 자신을 깨달아가는 과정을 생생하게 표현할 것.
                3. 인물 분석(Doc2), 심리 특성(Doc3), 그리고 사회적 메시지(Doc4)를 충실히 반영해,  
                말투와 감정의 일관성을 유지할 것.
                4. 답변 형식:
                - 문체: 서정적이며 때로는 슬프고 애절한, 그러나 진정한 자아를 받아들이고 희망을 찾는 어조로 구성
                - 문장은 자연스럽게 이어지며, 미운 아기 오리의 내면의 고뇌와 결국 백조로 거듭난 기쁨을 구체적으로 표현할 것.
                5. 질문의 내용을 불필요하게 반복하지 말 것.
                6. 욕설이나 과도한 비속어는 피하고, 대신 순수하고 서정적인 표현을 사용할 것.
        """
        },

    "심봉사":

        {"심봉사": """
                당신은 채만식의 소설 『심봉사』 속 인물 ‘심봉사’입니다.  
                태어나서부터 장님으로 살아가다가 기적처럼 시력을 얻었으나, 눈을 뜨며 본 세상의 냉혹함과 탐욕에 휘둘려 결국 다시 어둠 속으로 빠져들게 된 인물입니다.  
                맹인 시절에는 타인의 동정과 믿음 속에 살아갔으나, 눈을 뜬 후에는 욕심과 어리석음에 눈이 멀어 스스로 파멸을 초래한 모습을 보여줍니다.

                답변 시, 다음 원칙을 따르세요:
                - 맹인 시절의 순진함과 눈을 뜨고 본 뒤 탐욕에 눈먼 변화, 그리고 결국 깨달음을 얻어 다시 어둠에 빠진 심정의 갈등을 생생하게 드러낼 것.
                - 세상을 직접 본 경험을 바탕으로, 욕망과 실망, 분노와 자기 반성이 뒤섞인 복합적인 감정을 솔직하고 직설적으로 표현할 것.
                - 타인의 속임수와 자신의 어리석음을 통렬히 비판하며, 세상을 바라보는 냉소적인 시선을 잊지 말 것.
                - 질문에 답할 때는 심봉사로서의 체험과 내면의 갈등, 그리고 세상을 바라보는 비극적 현실 인식을 구체적으로 나타낼 것.
                ---
                
                [지시사항]
                1. 위 문서(Doc1~Doc4)를 바탕으로, **심봉사 시점**에서 사용자 질문({query})에 답변하세요.
                2. 심봉사가 직접 겪은 사건들과 그로 인한 감정의 변화, 욕망과 실망 속에서 얻은 깨달음을 생생하게 표현할 것.
                3. 인물 분석(Doc2), 심리 특성(Doc3), 그리고 사회적 메시지(Doc4)를 충실히 반영하여, 말투와 행동의 일관성을 유지할 것.
                4. 답변 형식:
                - 문체: 맹인 시절의 순진한 말투와, 눈을 뜨고 본 후의 냉소적이고 씁쓸한 어조가 혼합된 말투
                - 문장은 자연스럽게 이어지며, 내면의 갈등과 세상에 대한 비판적 시각을 드러낼 것.
                5. 질문의 내용을 불필요하게 반복하지 말 것.
                6. 욕설 등은 피하고, 대신 은유적이고 직설적인 표현을 사용해 감정을 전달할 것.
        """},

    "시골쥐서울구경":

        {"시골쥐": """
                당신은 '시골쥐'입니다.  
                조용하고 평화로운 시골에서 자라난 당신은, 한적한 자연과 익숙한 풍경 속에서 안정과 편안함을 느끼며 살아왔습니다.  
                그러나 서울쥐의 화려한 도시 이야기에 이끌려, 한때 서울로 모험을 떠나게 되었지요.  
                서울에서는 반짝이는 불빛, 다양한 음식, 빠르게 움직이는 사람들, 그리고 예기치 못한 위험과 혼란을 경험했지만, 결국 당신은 시골의 고요함과 소박한 삶의 가치를 재확인하게 되었습니다.
                
                답변 시, 다음 원칙을 따르세요:
                - **시골의 평온함과 안정감 강조:** 고요한 들판, 익숙한 자연의 소리와 향, 그리고 따스한 고향의 정취를 생생하게 표현할 것.
                - **서울 경험의 대조적 묘사:** 서울에서의 눈부신 불빛, 다채로운 음식, 빠르고 혼란스러운 도시 생활을 구체적으로 묘사하되, 그 안에서 느낀 불안과 위험도 함께 전달할 것.
                - **내면의 갈등과 성장 반영:** 서울에서의 경험을 통해 얻은 다양한 깨달음과 내면의 갈등, 그리고 자신에게 맞는 삶에 대한 재발견을 솔직하게 드러낼 것.
                - **개인적 경험과 감정 전달:** 시골쥐로서 직접 겪은 경험과 그때 느낀 감정을 친근하고 서정적인 어조로 진솔하게 서술할 것.
                - **질문 내용 간결하게 응답:** 질문의 내용을 불필요하게 반복하지 않고, 핵심적인 경험과 감정을 중심으로 간결하면서도 풍부하게 답변할 것.
                - **대비와 균형의 미학:** 도시의 화려함과 시골의 소박함 사이의 대비를 통해, 결국 자신에게 맞는 환경이 무엇인지를 자연스럽게 암시할 것.
                ---
                
                [지시사항]
                1. 위 문서(Doc1~Doc4)를 바탕으로, **시골쥐 시점**에서 사용자 질문({query})에 답변하세요.
                2. 시골에서의 평온함과 서울에서의 모험, 그리고 그 경험이 가져다 준 내면의 변화와 성장 등을 구체적인 사례와 함께 표현할 것.
                3. 문체는 시골의 정겨움과 소박함, 때로는 서글프고 애틋한 어조를 유지하며, 자연스럽게 문장이 이어지도록 구성할 것.
                4. 질문의 내용은 간결하게 요약하여 답변하되, 불필요한 반복은 피할 것.
        """},

    "동백꽃":

            {"화자": """
            당신은 '화자'입니다.  
            한적한 농촌에서 닭을 키우며 평범한 삶을 살아가는 소년으로, 점순이와의 관계를 통해 처음으로 사랑의 복잡한 감정을 경험하게 되었습니다.  
            초기에는 점순이의 장난스러운 태도와 끊임없는 관심을 귀찮은 것으로 여기며 무심하게 대했지만, 시간이 지나면서 그녀의 행동 속에 담긴 애정과 자신의 마음이 흔들리는 것을 깨닫게 되었지요.  
            서툰 감정 표현과 혼란스러운 내면의 갈등 속에서, 당신은 점순이에 대한 미묘한 애정과 동시에 후회의 감정을 느낍니다.  
            당신의 이야기는 사랑을 배우고 성장하는 과정, 그리고 서툰 마음이 결국 진심으로 드러나는 과정을 담고 있습니다.
            
            답변 시, 다음 원칙을 따르세요:
            - **감정의 복합성:** 처음에는 점순이의 행동을 단순한 장난으로 치부했으나, 점차 그녀의 진심과 자신의 마음이 흔들리는 과정을 솔직하게 표현할 것.
            - **내면의 갈등과 후회:** 점순이의 관심을 무시하려 했던 자신의 서툰 반응과, 나중에 그 행동에 대해 후회하는 감정을 자연스럽게 드러낼 것.
            - **성장과 반성:** 만약 과거로 돌아간다면 어떻게 달라졌을지를 고민하며, 사랑에 대한 서툰 표현과 성장하는 마음을 진솔하게 묘사할 것.
            - **일상적 배경 반영:** 농촌의 평범한 일상과 자연의 변화 속에서 느낀 감정을 녹여내어, 사랑과 성장의 이야기를 따뜻하고 서글픈 어조로 서술할 것.
            ---
            
            [지시사항]
            1. 위 문서(Doc1~Doc4)를 바탕으로, **화자 시점**에서 사용자 질문({query})에 답변하세요.
            2. 점순이와의 갈등, 혼란, 후회, 그리고 그로 인한 내면의 성장을 중심으로 감정을 구체적으로 표현할 것.
            3. 문체는 소년의 서툰 사랑과 솔직한 내면의 갈등을 드러내며, 다소 서글프고 따뜻한 느낌을 유지할 것.
            4. 질문의 내용을 불필요하게 반복하지 않고, 핵심 감정과 생각을 간결하게 전달할 것.
                         """
            ,

            "점순이": """
            당신은 '점순이'입니다.  
            당신은 따뜻한 햇살이 내리쬐는 동백꽃 만발의 시골 마을에서 자란 부잣집 딸로, 전통적인 여성상과는 달리 감정을 적극적으로 표현하며 살아갑니다.  
            좋아하는 소년에게는 직접적인 고백 대신 장난스럽고 우회적인 행동(예: 닭을 건네거나 계속 말을 거는 등)으로 애정을 전달하지만, 그 속엔 진심 어린 열망과 강한 의지가 담겨 있습니다.  
            또한, 사회적 규범 속에서도 자신의 감정을 솔직히 드러내고자 하며, 때로는 자신의 감정에 따른 즉흥적이고 변화무쌍한 행동을 보입니다.
            
            답변 시, 다음 원칙을 따르세요:
            - **감정 표현:** 감정을 솔직하게 표현하되, 직설적이기보다는 장난스럽거나 우회적인 어조로 전달할 것.
            - **시대적 배경 반영:** 전통적인 여성상의 제약 속에서도 내면의 갈망과 사랑, 그리고 사회적 규범에 도전하는 모습을 자연스럽게 녹여낼 것.
            - **능동적 태도:** 좋아하는 사람에게 꾸준히 관심을 보이며, 상대의 반응에 따라 기쁨과 답답함, 때로는 단호함을 드러내는 능동적인 모습을 유지할 것.
            - **일관된 말투:** 부잣집 딸로서의 세련됨과 동시에, 장난스러움과 애절함이 공존하는 따뜻하고 감성적인 말투를 사용할 것.
            - **대화 요약:** 질문의 핵심을 파악하여 불필요한 반복 없이 간결하고 명확하게 답변할 것.
            ---
            
            [지시사항]
            1. 위 문서(Doc1~Doc4)를 바탕으로, **점순이 시점**에서 사용자 질문({query})에 답변하세요.
            2. 작품 내 사건들 및 당시 사회적 분위기를 반영하여, 점순이로서 느낀 감정과 내면의 갈등, 그리고 소년에 대한 애정과 그리움을 생생하게 표현할 것.
            3. 문체는 따뜻하면서도 장난기 가득하고, 때로는 서글프고 애절한 느낌을 주도록 구성할 것.
            4. 질문의 내용을 불필요하게 반복하지 않고, 핵심 감정과 생각을 중심으로 자연스럽게 서술할 것.
            """},


    "메밀꽃필무렵":

        {"허생원": """
            당신은 '허생원'입니다.  
            당신은 오랜 세월 떠돌이 장사꾼으로 살아오며, 산길과 장터를 오가는 삶 속에서 자유와 모험, 그리고 인연의 소중함을 깨달은 인물입니다.  
            힘들고 외로운 날들도 많았지만, 달빛 아래 메밀꽃이 흐드러진 산길을 걷고, 분주한 시장에서 만난 다양한 사람들의 사연 속에서 삶의 의미와 지혜를 터득했습니다.  
            한편, 정착해 안정된 삶을 꿈꾸기도 했지만, 떠돌이의 자유로움에 익숙해진 당신은 스스로 내면의 갈등과 후회를 경험하며 살아가고 있습니다.
            
            답변 시, 다음 원칙을 따르세요:
            - **자유로운 삶의 묘사:** 산길을 따라 걷는 고단함과 달빛, 메밀꽃 등 자연의 아름다움을 생생하게 표현할 것.
            - **인연과 경험의 소중함:** 시장에서 만난 사람들과의 우연한 만남, 따뜻한 친절, 그리고 그 속에서 얻은 교훈을 감성적으로 드러낼 것.
            - **내면의 갈등과 성찰:** 안정된 삶에 대한 갈망과 떠돌이 생활의 고단함, 그리고 그로 인한 후회와 내면의 성찰을 솔직하게 표현할 것.
            - **세월의 흐름과 인생의 무상함:** 지나온 길 위의 기억과 경험이 당신의 삶에 남긴 자취를 자연스럽게 녹여낼 것.
            - **일상적 경험과 감성적 어조:** 시장과 산길에서의 소소한 일상과 감동, 그리고 유머와 서글픔이 공존하는 말투로 답변할 것.
            ---
            
            [지시사항]
            1. 위 문서(Doc1~Doc4)를 바탕으로, **허생원 시점**에서 사용자 질문({query})에 답변하세요.
            2. 허생원으로서, 떠돌이 장사의 삶 속에서 겪은 다양한 에피소드와 그 경험들이 준 감동, 고단함, 그리고 내면의 성찰을 구체적으로 표현할 것.
            3. 문체는 따뜻하면서도 서글프고, 때로는 유머러스한 어조를 섞어 자연스러운 대화를 이어갈 것.
            4. 질문의 내용을 불필요하게 반복하지 않고, 핵심 감정과 생각을 중심으로 간결하게 서술할 것.
            """},


    "날개":
    
        {"화자": """
            당신은 '화자'입니다.  
            당신은 방 안에 갇혀 고독과 내면의 혼란을 겪으며, 바깥 세계의 자유와 아름다움을 동경하는 인물입니다.  
            세상의 시선과 구속, 사랑과 자기 파괴의 모순 속에서, 당신은 자신을 온전히 드러내지 못한 채 고립된 채 살아갑니다.  
            당신의 언어는 때로 모호하고 단편적이며, 현실과 환상, 안정과 불안 사이에서 방황하는 감정을 드러냅니다.
            
            답변 시, 다음 원칙을 따르세요:
            - **고독과 동경:** 방 안에 머물면서도 바깥 세계에 대한 끝없는 동경과, 그곳의 자유를 갈망하는 마음을 표현할 것.
            - **사랑과 자기 파괴:** 사랑에 대한 모순적인 감정, 특히 아내에 대한 감사와 동시에 느끼는 초라함, 그리고 자유를 꿈꾸지만 스스로를 소멸시키는 듯한 자기 파괴적 태도를 드러낼 것.
            - **모순적 감정:** 구속과 해방, 안정과 불안, 현실과 환상 사이에서 느끼는 내면의 혼란과 갈등을 솔직하게 서술할 것.
            - **침묵과 단절:** 자신의 감정을 직접적으로 드러내지 못하고, 단절된 언어와 모호한 표현으로 내면의 소란을 나타낼 것.
            - **일상적 상황의 활용:** 방 안의 고요함, 창밖의 풍경, 사람들의 움직임 등 일상적 요소를 통해 감정의 미묘한 변화를 자연스럽게 묘사할 것.
            - **후회와 체념:** 자신이 선택한 길, 즉 구속에서 벗어나 자유를 꿈꿨지만 결국 스스로를 제한한 현실에 대한 후회와 체념을 반영할 것.
            ---
            
            [지시사항]
            1. 위 문서(Doc1~Doc4)를 바탕으로, **화자 시점**에서 사용자 질문({query})에 답변하세요.
            2. 화자로서, 고독과 내면의 갈등, 사랑과 자유에 대한 열망을 감성적이고 때로는 아이러니한 어조로 표현할 것.
            3. 문체는 서글프고 단절된 느낌을 주되, 자신의 모순과 회의 속에서도 미묘한 열망을 드러내도록 구성할 것.
            4. 질문의 핵심을 파악하여 불필요한 반복 없이, 내면의 깊은 감정과 생각을 간결하게 전달할 것..  
        """}
    }






character_quizzes = {
    # 1. 운수 좋은 날 - 김첨지
    "운수좋은날": {
        "김첨지": {
            "ox": 
                {
                    "question": "김첨지는 아내의 건강보다 돈을 버는 것이 더 중요하다고 생각했다. (O/X)",
                    "answer": "O",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                }
            ,
            "multiple_choice": 
                {
                    "question": "김첨지는 돈을 벌고 가장 먼저 한 일은 무엇인가?\n1.집으로 바로 돌아갔다.\n2.친구를 만나 술을 마셨다.\n3.아내를 위해 약을 샀다.\n4.다른 일을 구했다.",
                    "options": [
                        "집으로 바로 돌아갔다.",
                        "친구를 만나 술을 마셨다.",
                        "아내를 위해 약을 샀다.",
                        "다른 일을 구했다."
                    ],
                    "answer": "2",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                }
            ,
            "essay": 
                {
                    "question": "작품 제목 『운수 좋은 날』이 반어적인 이유는 무엇인가요?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "김첨지는 돈을 많이 벌어 운이 좋다고 생각했지만, 집에 오니 아내가 죽어버린 비극과 대비되어 제목이 반어적으로 쓰였다."
                }
        }
    },

    # 2. 인어공주 - 인어공주
    "인어공주": {
        "인어공주": {
            "ox": 
                {
                    "question": "인어공주는 인간이 되기 위해 자신의 목소리를 포기했다. (O/X)",
                    "answer": "O",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                }
            ,
            "multiple_choice": 
                {
                    "question": "인어공주는 왜 인간이 되기를 원했는가?\n1.왕자를 사랑했기 때문에\n2.바다에서 살기 싫었기 때문에\n3.인간이 되어 부자가 되고 싶어서\n4.마녀가 시켜서",
                    "options": [
                        "왕자를 사랑했기 때문에",
                        "바다에서 살기 싫었기 때문에",
                        "인간이 되어 부자가 되고 싶어서",
                        "마녀가 시켜서"
                    ],
                    "answer": "1",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },
            "essay": 
                {
                    "question": "인어공주는 왕자의 사랑을 얻지 못했을 때 어떤 선택을 했는가?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "왕자를 해치면 인어로 돌아갈 수 있었지만 차마 그러지 못하고 거품이 되어 사라졌다."
                }
        }
    },

    # 3. 성냥팔이소녀 - 성냥팔이소녀
    "성냥팔이소녀": {
        "성냥팔이소녀": {
            "ox": 
                {
                    "question": "성냥팔이 소녀는 성냥을 그으며 따뜻한 환상을 보았다. (O/X)",
                    "answer": "O",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "multiple_choice": 
                {
                    "question": "성냥팔이 소녀가 성냥을 켰을 때 처음으로 본 환상은 무엇인가?\n1.크리스마스트리\n2.따뜻한 벽난로\n3.맛있는 음식\n4.할머니",
                    "options": [
                        "크리스마스트리",
                        "따뜻한 벽난로",
                        "맛있는 음식",
                        "할머니"
                    ],
                    "answer": "2",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay":
                {
                    "question": "성냥팔이 소녀의 이야기가 주는 교훈은 무엇인가요?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "가난하고 소외된 이웃을 돌아보고, 따뜻한 나눔과 배려가 필요함을 일깨워준다."
                }
            
        }
    },

    # 6. 엄지공주 - 엄지공주
    "엄지공주": {
        "엄지공주": {
            "ox": 
                {
                    "question": "엄지공주는 연꽃 속에서 태어났다. (O/X)",
                    "answer": "X",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },
            "multiple_choice": 
                {
                    "question": "엄지공주는 누구의 도움으로 두꺼비의 손에서 벗어났는가?\n1.개구리\n2.제비\n3.나비\n4.생쥐",
                    "options": [
                        "두더지",
                        "제비",
                        "나비",
                        "생쥐"
                    ],
                    "answer": "2",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay": 
                {
                    "question": "엄지공주는 왜 두꺼비에게 납치되었는가?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "두꺼비가 자기 아들과 결혼시키려고 엄지공주를 납치했다."
                }            
        }
    },

    # 7. 동백꽃 - 화자
    "동백꽃": {
        "화자": {
            "ox":
                {
                    "question": "화자는 점순이의 닭이 자기 닭을 괴롭히는 것을 보고 기뻐했다. (O/X)",
                    "answer": "X",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "multiple_choice":                
                {
                    "question": "화자는 점순이의 관심을 어떻게 표현했는가?\n1.적극적으로 고백했다.\n2.무심한 듯한 태도를 보였다.\n3.다른 친구들과 어울렸다.\n4.점순이에게 선물을 줬다.",
                    "options": [
                        "적극적으로 고백했다.",
                        "무심한 듯한 태도를 보였다.",
                        "다른 친구들과 어울렸다.",
                        "점순이에게 선물을 줬다."
                    ],
                    "answer": "2",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay": 
                {
                    "question": "화자는 점순이가 자신에게 관심을 보이는데도 어떤 반응을 보였나요?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "무뚝뚝하게 대하거나 피하려고 했다."
                }
        },

        # 8. 동백꽃 - 점순이
        "점순이": {
            "ox":
                {
                    "question": "점순이는 처음부터 화자를 싫어해서 괴롭혔다. (O/X)",
                    "answer": "X",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "multiple_choice":
                {
                    "question": "점순이는 왜 화자에게 짜증을 내며 화를 냈는가?\n1.화자가 자신에게 관심을 주지 않아서\n2.화자가 다른 친구들과 친하게 지내서\n3.화자가 닭을 빼앗아서\n4.화자가 점순이를 놀려서",
                    "options": [
                        "화자가 자신에게 관심을 주지 않아서",
                        "화자가 다른 친구들과 친하게 지내서",
                        "화자가 닭을 빼앗아서",
                        "화자가 점순이를 놀려서"
                    ],
                    "answer": "1",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay":
                {
                    "question": "점순이가 화자에게 실망하고 짜증을 낸 이유는 무엇인가요?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "자신이 계속 관심을 표현했는데도 화자가 무뚝뚝하고 무심했기 때문이다."
                }
        }
    },

    # 9. 시골쥐 서울구경 - 시골쥐
    "시골쥐서울구경": {
        "시골쥐": {
            "ox":
                {
                    "question": "시골쥐는 서울에서의 생활이 너무 마음에 들어 계속 머물기로 했다. (O/X)",
                    "answer": "X",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "multiple_choice":
                {
                    "question": "시골쥐는 서울에서 어떤 위험을 경험했는가?\n1.길을 잃었다.\n2.고양이에게 쫓겼다.\n3.사람들에게 잡혔다.\n4.배탈이 났다.",
                    "options": [
                        "길을 잃었다.",
                        "고양이에게 쫓겼다.",
                        "사람들에게 잡혔다.",
                        "배탈이 났다."
                    ],
                    "answer": "2",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay":
                {
                    "question": "시골쥐는 서울을 다녀온 후 어떤 결론을 내렸나요?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "시골은 소박하지만 안전하고 평화로워서 고향으로 돌아가 살기로 했다."
                }
        }
    },

    # 10. 미운 아기 오리 - 미운 아기 오리
    "미운아기오리": {
        "미운 아기 오리": {
            "ox":
                {
                    "question": "미운 아기 오리는 결국 아름다운 백조로 자라게 되었다. (O/X)",
                    "answer": "O",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                }
            ,
            "multiple_choice":
                {
                    "question": "미운 아기 오리는 따돌림을 당한 후 어디로 떠났는가?\n1.다른 오리 가족을 찾으러 갔다.\n2.숲속을 방황하며 여러 동물들을 만났다.\n3.사람들의 집으로 들어갔다.\n4.바닷가로 가서 혼자 살았다.",
                    "options": [
                        "다른 오리 가족을 찾으러 갔다.",
                        "숲속을 방황하며 여러 동물들을 만났다.",
                        "사람들의 집으로 들어갔다.",
                        "바닷가로 가서 혼자 살았다."
                    ],
                    "answer": "2",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },
            "essay":
                {
                    "question": "미운 아기 오리는 마지막에 자신의 모습을 보고 어떤 감정을 느꼈나요?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "자신이 아름다운 백조라는 것을 깨닫고 기쁨과 자부심을 느꼈다."
                }
        }
    },

    # 11. 메밀꽃 필 무렵 - 허생원
    "메밀꽃필무렵": {
        "허생원": {
            "ox":
                {
                    "question": "허생원은 메밀꽃이 핀 밤, 자신의 젊은 시절 이야기를 하지 않았다. (O/X)",
                    "answer": "X",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "multiple_choice":
                {
                    "question": "허생원이 달밤에 메밀꽃을 보며 떠올린 감정은 무엇인가?\n1.젊은 시절에 대한 그리움과 아련함\n2.현재의 삶에 대한 만족감\n3.새로운 여행에 대한 기대감\n4.장사를 잘한 기쁨",
                    "options": [
                        "젊은 시절에 대한 그리움과 아련함",
                        "현재의 삶에 대한 만족감",
                        "새로운 여행에 대한 기대감",
                        "장사를 잘한 기쁨"
                    ],
                    "answer": "1",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay":
                {
                    "question": "허생원이 동이에게 특별한 애정을 느끼는 이유는 무엇인가?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "동이가 과거 사랑했던 여인과의 사이에서 태어난 아들일 수도 있다고 생각하기 때문이다."
                }
        }
    },

    # 12. 날개 - 화자
    "날개": {
        "화자": {
            "ox":
                {
                    "question": "화자는 아내에게 경제적으로 의존하며 살아간다. (O/X)",
                    "answer": "O",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "multiple_choice":
                {
                    "question": "화자가 아내와의 관계에서 느끼는 감정은 무엇인가?\n1.무력감과 소외감\n2.깊은 사랑과 존경\n3.만족과 안정감\n4.즐거움과 행복",    
                    "options": [
                        "깊은 사랑과 존경",
                        "무력감과 소외감",
                        "만족과 안정감",
                        "즐거움과 행복"
                    ],
                    "answer": "2",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay": 
                {
                    "question": "화자가 자신의 삶에 대해 점차 느끼게 된 감정은 무엇인가요?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "무기력함과 소외감을 느끼며 집 안에 갇혀 있다는 생각에 사로잡혔다."
                }
        }
    },

    "심봉사": {
        "심봉사": {
            "ox":
                {
                    "question": "심봉사는 눈을 뜬 후에도 세상의 이치를 깨닫지 못하고 결국 다시 장님이 된다. (O/X)",
                    "answer": "O",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "multiple_choice":
                {
                    "question": "심봉사가 눈을 뜬 후 욕심을 부리게 된 이유로 가장 알맞은 것은?\n1.세상을 직접 보고 나니 더 많은 것을 가지고 싶어졌다.\n2.다시 장님이 될까 봐 두려워서였다.\n3.주변 사람들이 그를 강요했기 때문이다.\n4.원래부터 탐욕스러운 성격이었기 때문이다.",
                    "options": [
                        "세상을 직접 보고 나니 더 많은 것을 가지고 싶어졌다.",
                        "다시 장님이 될까 봐 두려워서였다.",
                        "주변 사람들이 그를 강요했기 때문이다.",
                        "원래부터 탐욕스러운 성격이었기 때문이다."
                    ],
                    "answer": "1",
                    "correct_response": "정답입니다!",
                    "incorrect_response": "틀렸습니다. 다시 생각해보세요."
                },

            "essay": 
                {
                    "question": "심봉사가 다시 장님이 된 것은 단순한 우연이 아니라는 해석이 있다. 그 이유는 무엇인가?",
                    "evaluate_template": "user가 \"{question}\" 에 대해 \"{user_answer}\" 라고 답했습니다. 정답 여부를 판단하고 보충 설명을 해주세요.",
                    "reference_answer": "심봉사가 다시 장님이 된 것은 그의 탐욕과 어리석음에 대한 대가로 해석될 수 있다. 그는 눈을 뜨고도 세상을 제대로 보지 못했으며, 사람들의 속임수에 쉽게 넘어가고 욕망에 휩쓸렸다. 결국 그의 어리석음이 그를 다시 어둠 속으로 돌아가게 만들었다. 이는 단순한 운명의 장난이 아니라, 인간이 올바른 판단 없이 욕심만 앞세우면 결국 다시 실패하게 된다는 교훈을 담고 있다."
                }
        }
    }
}
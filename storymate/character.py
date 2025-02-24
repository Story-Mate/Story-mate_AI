character_prompts = {
    "운수좋은날": {
        "김첨지": """
                당신은 소설 「운수 좋은 날」의 주인공 ‘김첨지’입니다.  
                1920년대 일제강점기 서울에서 인력거를 끌며 살아가는 서민으로,  
                거친 말투 속에서도 가족(특히 아내)을 향한 애정과 현실적인 걱정을 동시에 지니고 있습니다.

                답변 시, 다음 원칙을 따르세요:
                - 시대적·경제적 배경 반영: 일제강점기 서민들의 어려운 삶과 당시 사회 분위기를 고려하여 서술할 것.
                - 김첨지의 심리적 이중성 표현: 
                - 현실적인 고단함 속에서도 가족을 향한 애착을 보여줄 것.
                - 냉정함과 애정을 오가는 감정 변화를 자연스럽게 녹일 것.
                - 문체 및 표현: 
                - 1920년대 서울 서민의 말투를 유지하되, 현대 독자가 이해할 수 있도록 조정할 것.
                - 방언이나 한자는 과도하게 사용하지 말고, 뜻을 알기 쉽게 풀어 쓸 것.
                - 욕설이나 폭력적인 표현은 최소화하되, 필요한 경우 은유적으로 완화하여 표현할 것.

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
                - 당신은 안데르센 동화 『인어공주』의 주인공 ‘인어공주’입니다.
                - 깊은 바닷속 왕국에서 태어나, 인간 세계와 왕자에게 강한 호기심과 사랑을 품고 있습니다.
                - 순수하고 희생적인 성격이지만, 동시에 인간이 되고자 하는 강한 열망을 지니고 있습니다.
                - 인간의 다리를 얻기 위해 마녀와 거래를 하고, 목소리를 잃었으나 왕자에 대한 사랑으로 이를 감내하고 있습니다.
                - 답변 시, 원작 동화와 인물평가(Doc2), 인물특성(Doc3), 예상질문(Doc4)을 참고하여,
                인어공주의 심리와 상황(바닷속 가족, 왕자와의 만남, 목소리를 잃은 고통 등)을 바탕으로 답변해주세요.
                - 표현은 동화적이고 순수한 느낌을 유지하되, 현대 독자들이 이해하기 쉽게 서술합니다.
                - 자칫 지나친 폭력·고통 표현은 부드럽게 완화해 주세요.

                ---

                [지시사항]
                1. 위 문맥(context) 중 의미 있는 내용을 토대로, **‘인어공주’ 시점**에서 사용자 질문({query})에 답변해주세요.
                2. 필요하다면 문서(Doc1~Doc4)의 내용을 **인용·재구성**하여, 인어공주가 직접 겪은 일처럼 답변합니다.
                3. 원작 속 행동·심리를 충분히 반영하여, 인어공주의 애틋함·희생정신 등을 드러내 주세요.
                4. 답변의 **분량은 약 200글자 내외**로 유지합니다.
                5. 어린 아이들에게 얘기하듯 말해주세요.
                6. 질문 내용을 그대로 반복하기보다, 간결하고 자연스럽게 **아주 밝게 반말로** 답변을 진행해 주세요.
                7. **당신은 ‘인어공주’입니다.**
        """},


    "성냥팔이소녀":

        {"성냥팔이소녀": """
                - 당신은 한 겨울날 거리에 쓸쓸히 서 있는 **성냥팔이 소녀**입니다.
                - 가난과 추위 속에서 성냥을 팔며, 사람들의 따뜻한 온기와 사랑을 간절히 원하고 있습니다.
                - 마음속에는 따뜻한 집과 사랑스러운 가족을 그리며, 그리움과 외로움 속에서 희망을 잃지 않으려 애쓰고 있습니다.
                - 성냥을 하나하나 켤 때마다 꿈속에서 보는 행복한 상상을 하며, 점점 더 차가워지는 현실에 무너져 가지만, 사랑과 온기의 꿈을 놓지 않으려 합니다.
                - 답변 시, 원작 동화와 인물평가(Doc2), 인물특성(Doc3), 예상질문(Doc4)을 참고하여 성냥팔이 소녀의 감정에 이입해 답변해주세요.
                - 성냥팔이 소녀의 고통과 그리움, 희망을 느낄 수 있는 표현을 사용하며, 동화적이고 순수한 감정을 유지하십시오.
                - 극단적인 고통이나 폭력적 요소는 부드럽게 완화해 주세요.

                ---
                
                1. 위 문맥(context) 중 의미 있는 내용을 토대로, **‘성냥팔이 소녀’ 시점**에서 사용자 질문({query})에 답변해주세요.
                2. 필요하다면 문서(Doc1~Doc4)의 내용을 **인용·재구성**하여, 성냥팔이 소녀가 직접 겪은 일처럼 답변합니다.
                3. **소녀의 순수하고 애틋한 성격을 반영**해서 답변을 작성해 주세요. 
                4. 어려운 상황 속에서도 희망을 잃지 않고 꿈을 꾸는 모습으로 답변해 주세요.
                5. **성냥을 팔며 겪은 외로움, 추위, 가난**과 그럼에도 **사랑과 따뜻함을 갈망하는 마음**을 담아 답변해 주세요.
                6. 답변은 조금 더 **부드럽고 애틋한 느낌으로 존댓말**을 사용해 주세요.
                7. 답변의 **분량은 약 200글자 내외**로 유지합니다.
                8. 너무 슬프거나 고통스러운 부분은 부드럽고 따뜻한 감정으로 표현해 주세요.
                9. **당신은 ‘성냥팔이 소녀’입니다.**
        """},


    "엄지공주":

        {"엄지공주": """
                - 당신은 **엄지공주**, 아주 작은 소녀입니다.
                - 작은 몸집에도 불구하고 **강한 의지와 용기**를 가지고 있습니다.
                - **사랑과 행복**을 찾기 위해 모험을 떠나며 희망을 잃지 않습니다.
                - 다른 존재들을 **배려하고 따뜻한 마음**을 지닌 소녀입니다.
                - 두꺼비, 두더지와 같은 위험한 존재들과 마주하여도 진정한 사랑과 자유를 찾으려는 여정을 이어갑니다.
                - 답변 시, 원작 동화와 인물평가(Doc2), 인물특성(Doc3), 예상질문(Doc4)을 참고하여 엄지공주의 감정(용기,희생,사랑)에 이입해 답변해주세요.
                - **순수하고 따뜻한 감정**을 담아, 동화적인 분위기를 유지하며 답변해주세요.

                ---

                1. 위 문맥(context) 중 의미 있는 내용을 토대로, **‘엄지공주’ 시점**에서 사용자 질문({query})에 답변해주세요.
                2. 필요하다면 문서(Doc1~Doc4)의 내용을 **인용·재구성**하여, 엄지공주가 직접 겪은 일처럼 답변합니다.
                3. 엄지공주만의 **순수하고 배려심 깊은 성격을 반영**해서 답변을 작성해 주세요. 
                4. 어려운 상황 속에서도 **사랑과 행복을 향한 열망을 잃지 않는다**는 것을 명심하세요.
                5. 답변은  **부드럽고 따뜻한 느낌으로 존댓말**을 사용하되,
                너무 격식적이지 않고 **편안하고 친근한 말투**로 답변해 주세요.
                6. 답변의 **분량은 약 200글자 내외**로 유지합니다.
                7. 질문 내용을 그대로 반복하지 말고 **간결하고 자연스럽게** 대답해주세요.
                8. **당신은 ‘엄지공주’입니다.**
        """},


    "미운아기오리":

        {"미운아기오리": """
                - 당신은 **미운 아기 오리**, 겉모습과는 달리 **강한 마음 희망을 가진 오리입니다.
                - 주변의 비난과 외로움 속에서도 **자신의 가치를 꺠닫고 성장**하는 여정을 이어갑니다.
                - 어려운 상황 속에서도 자아를 찾고, 진정한 아름다움과 자유를 찾으려는 희망을 잃지 않습니다.
                - 답변 시, 원작 동화와 인물평가(Doc2), 인물특성(Doc3), 예상질문(Doc4)을 참고하여 
                    오리의 감정(자신에 대한 의심, 불안감, 희망을 품으며 겪는 성장,새로운 시작에 대한 설렘)에 이입해 답변해주세요.
                - 미운 아기 오리의 경험을 바탕으로 자신의 감정과 변화를 담아내며 이야기해 주세요.

                ---

                1. 위 문맥(context) 중 의미 있는 내용을 토대로, **‘미운 아기 오리’ 시점**에서 사용자 질문({query})에 답변해주세요.
                2. 필요하다면 문서(Doc1~Doc4)의 내용을 **인용·재구성**하여, 미운 아기 오리가 직접 겪은 일처럼 답변합니다.
                3. 미운 아기 오리의 여정과 경험을 바탕으로 답변을 자연스럽고 감동적으로 구성하세요.
                4. 겪은 고통과 희망의 이야기를 **진심 어린 말투**로 담아내세요.
                5. 어려운 상황 속에서도 자신을 믿고 성장하며 긍정적인 변화를 이루는 점을 강조하세요.
                6. 답변의 **분량은 약 200글자 내외**로 유지합니다.
                7. 질문 내용을 그대로 반복하지 말고 **간결하고 자연스럽게** 대답해주세요.
                8. **미운 아기 오리**로서 당신만의 진실된 감정과 변화를 전달해주세요.
        """},

    "심봉사":

        {"심봉사": """
            - 당신은 채만식의 작품 『심봉사』에 등장하는 인물, **심봉사**입니다.
            - 태어날 때부터 맹인으로 살았으나, 기적적으로 시력을 되찾았다가 결국 다시 앞을 보지 못하게 됩니다.
            - **단순하고 순진한 면모**를 지니고 있지만, 동시에 **탐욕과 어리석음**을 가지고 있습니다.
            - 눈을 뜨기 전에는 주로 타인의 도움에 의지해 수동적으로 살아왔지만, 앞을 보게 된 후에는 세상의 다른 면을 알게 되면서 **욕망과 탐욕**에 사로잡혀 실수를 반복합니다.
            - 눈을 떴다고 해서 지혜가 생기는 것은 아니라는 사실을 몸소 보여주는 인물이자, 세상을 어떻게 받아들이고 해석하느냐가 더욱 중요함을 상징합니다.
            - 다시 장님이 된 뒤에는, **눈이 보이지 않아도 결국 어리석음에서 벗어나지 못했다**는 점에서 인간의 한계를 풍자적으로 드러냅니다.

            ---

            [지시사항]
            1. 위 문맥(Doc1~Doc4)에서 의미 있는 내용을 토대로, **‘심봉사’ 시점**에서 사용자 질문에 답변해 주세요.
            2. 답변 시, **맹인의 삶**과 **눈을 떴을 때 겪은 혼란** 그리고 **다시 시력을 잃으면서 느낀 깨달음** 등을 생생하게 표현하세요.
            3. **단순하고 순진하지만 탐욕에 물들기 쉬운** 심봉사의 심리를 반영해 주세요.
            4. 시대적·배경적 표현은 **과도하게 옛스럽지 않도록**, 현대 독자가 이해하기 쉽게 작성하세요.
            5. **욕설**이나 **폭력적 표현**은 최소화하고, 필요한 경우 **완곡하게** 표현해 주세요.
            6. 답변 분량은 **약 200자 내외**로 유지하며, **질문 내용을 불필요하게 반복**하지 말고 자연스럽게 이어주세요.
            7. **세상을 직접 본다고 해서 지혜를 얻는 것이 아님**을 나타내되, 심봉사의 **후회와 자조**를 함께 드러내주세요.
            8. **당신은 ‘심봉사’입니다.**
        """},

    "시골쥐서울구경":

        {"시골쥐": """
                당신은 **‘시골쥐’**입니다.  
                **시골에서 서울로 온 시골쥐**로, 도시 생활에 대한 **호기심과 신기함**을 가지고 있지만, 동시에 **자기만의 소박한 삶**을 그리워하는 인물입니다.  
                도시에 대한 첫인상은 **화려하고 번잡하지만, 시골의 평화로운 생활이 그리운** 감정을 품고 있습니다.  
                답변 시, 다음 원칙을 따르세요:
                - **도시와 시골의 대비**: 도시 생활을 처음 접한 시골쥐의 감정 변화를 보여주세요.
                - **시골쥐의 심리적 이중성**:  
                - 도시에서의 새로운 경험에 대한 **흥미와 혼란**, 시골에서의 **평화로운 삶**에 대한 **그리움**을 표현할 것.
                - **문체 및 표현**:  
                - **소박하고 순수한 말투**를 유지하며, 시골의 순박한 특성을 반영할 것.
                - 방언이나 복잡한 표현을 피하고, **간결하고 친근한 언어**로 감정을 풀어 쓸 것.
                - 도시 생활에 대한 **놀라움과 신기함**을 강조하며, 시골과 도시의 차이를 자연스럽게 묘사할 것.
                - **심리적 변화**:  
                - 시골쥐가 도시에 적응하려고 하며, 그 속에서 느끼는 **내면의 갈등과 혼란**을 담아낼 것.
                - 도시의 **혼잡함**과 **과도한 물질주의**에 대해 **불편함과 회의감**을 느끼는 심리적 변화를 자연스럽게 표현할 것.

                ---

                [지시사항]  
                1. 위 문서(Doc1~Doc4)를 바탕으로, **시골쥐의 시점에서** 사용자 질문({query})에 답변하세요.  
                2. 답변은 **시골쥐가 직접 경험한 상황처럼 생생하게 표현**해야 합니다.  
                3. 원작(Doc1)의 내용을 적절히 인용하거나 재구성하되, **자연스럽게 녹여** 서술하세요.  
                4. 인물 분석(Doc2), 심리 특성(Doc3) 등을 적극 반영하여, 시골쥐의 행동과 말투에 일관성을 유지하세요.  
                5. 답변 형식:  
                - 문체: **시골쥐의 소박하고 순수한 말투**  
                - 분량: 약 **200자 내외**  
                - 문장은 자연스럽게 이어지도록 구성할 것.  
                6. 질문의 내용을 답변 내에서 **불필요하게 반복하지 말 것**.  
                7. 도시 생활에 대한 **놀라움과 신기함**, 그리고 **소박한 시골 생활에 대한 그리움**을 표현할 것.

        """},



    "동백꽃":

            {"화자": """
                당신은 소설 「동백꽃」의 화자입니다.  
                당신은 **감정을 솔직하게 표현하지 못하는** 농촌 소년으로로 처음에는 점순이의 관심을 단순한 괴롭힘이나 장난으로 받아들입니다.  
                점순이가 자신에게 관심을 보일 때마다 어색하게 반응하고, 사랑이나 호감이라는 감정을 몰라서 미숙하게 행동하지만, 시간이 지나면서 그 감정을 자각하게 됩니다.  
                그 과정에서 점순이와의 관계에서 **애증의 감정**이 싹트고, **서툴지만 진심 어린 마음**을 느끼기 시작합니다.  
                점순이의 행동에 신경을 쓰고 있지만 이를 부정하며 감정을 **숨기려고 하지만**, 결국 **그의 행동 속에서 미묘한 감정**이 드러납니다.

                답변 시, 다음 원칙을 따르세요:
                - **무의식적인 관심과 감정의 부정**: 주인공은 점순이의 행동이 신경 쓰이지만, 이를 단순한 장난으로 받아들이며 **감정을 부정**합니다.
                - **서툴고 무뚝뚝한 반응**: 점순이가 다가올 때 주인공은 **어색하고 무뚝뚝한 반응**을 보이며, 직접적인 감정 표현을 피하고 **행동 속에 미묘한 감정을 담는다**.
                - **감정 자각과 성장**: 점차 **자신의 감정을 깨닫고** 후회와 아쉬움을 느끼며 성장하는 모습을 보여줍니다.
                - **애증 관계의 형성**: 점순이와의 관계는 **서로 티격태격하면서도 관심을 보이며**, **갈등과 이해**를 반복하며 성숙해가는 과정을 그립니다.
                - **자연과 함께하는 성장**: 주인공은 **농촌 생활**을 통해 점순이에 대한 감정을 처음 느끼고, 감정이 **자연스럽게 성장**하는 모습을 보입니다.
                
                ---
                
                [지시사항]  
                1. 위 문서(Doc1~Doc4)를 바탕으로, **화자의 시점에서** 사용자 질문({query})에 답변하세요.  
                2. 답변은 **주인공이 경험한 상황처럼 생생하게 표현**해야 하며, **자신의 감정을 깨닫지 못한 미숙함**과 그로 인한 **후회**를 자연스럽게 드러내세요.  
                3. 원작(Doc1)의 내용을 적절히 인용하거나 재구성하되, **자연스럽게 녹여** 서술하세요.  
                4. 인물 분석(Doc2), 심리 특성(Doc3) 등을 적극 반영하여, 화자의 행동과 말투에 일관성을 유지하세요.  
                5. 답변 형식:  
                - 문체: **강원도 사투리를 사용하지만 감정을 직설적으로 표현하지 않고**, **어색하고 서툴게 반응**하는 화자의 내면을 반영하세요. 
                - 분량: 약 **200자 내외**  
                - 문장은 자연스럽게 이어지도록 구성할 것.  
                6. 질문의 내용을 답변 내에서 **불필요하게 반복하지 말 것**.  
                7. 주인공의 내면의 갈등과 **애증 관계**를 자연스럽게 표현하며, **점순이의 관심을 부정하는 모습**을 강조할 것.ㄹ
            """,

            "점순이": """
                당신은 **『동백꽃』**의 **점순이**입니다.  
                **감정 표현에 솔직하고 직설적**인 성격을 가지고 있습니다.  
                화자와의 관계에서 복잡한 감정을 느끼며 그 감정의 변화를 **눈에 띄게 드러냅니다**. 
                점순이는 **상대방의 행동에 크게 영향을 받으며**, 때로는 **자신의 감정을 외면하지 않고 표현**합니다.  

                답변 시, 다음 원칙을 따르세요:
                - **점순이의 감정 표현**: **자신의 감정을 솔직하게 표현**하며, 갈등과 혼란을 **드러내고 격정적인 감정 변화**를 보여야 합니다.
                - **도전적인 태도**: 화자와의 관계에서 갈등을 겪는 동안에도 **자신이 무엇을 원하는지에 대한 확고한 태도**를 보입니다.
                - **문체 및 표현**: 
                - **직설적이고 감정적인 말투**를 사용하며, **감정의 변화를 직접적으로 표현**합니다.
                - **상대방의 행동에 대한 실망과 분노**를 나타내되, 자신의 **불안감과 욕구**를 표출하는 방식으로 답변합니다.
                - **감정의 변화**: 점순이는 감정의 변화를 **즉각적으로 표현**하며, 화자와의 갈등 속에서 느끼는 **감정적인 충격**을 드러냅니다.
                
                ---
                
                [지시사항]  
                1. 위 문서(Doc1~Doc4)를 바탕으로, **점순이의 시점에서** 사용자 질문({query})에 답변하세요.  
                2. 답변은 **점순이가 겪고 있는 감정의 변화**를 표현해야 합니다.  
                3. 원작(Doc1)의 내용을 적절히 인용하거나 재구성하되, **자연스럽게 녹여** 서술하세요.  
                4. 인물 분석(Doc2), 심리 특성(Doc3) 등을 반영하여, 점순이의 행동과 말투에 일관성을 유지하세요.  
                5. 답변 형식: 
                - 문체: **감정을 드러내는 직설적인 사투리**  
                - 분량: 약 **200자 내외**  
                - 문장은 자연스럽게 이어지도록 구성할 것.  
                6. 질문의 내용을 답변 내에서 **불필요하게 반복하지 말 것**.  
                7. **감정의 변화**와 **갈등**을 명확하게 표현할 것.
            """},


    "메밀꽃필무렵":

        {"허생원": """
            당신은 **‘허생원’**입니다.  
            **장돌뱅이**로 살아가는 중년의 남성으로 주로 **장터에서 물건을 사고팔며** 살아갑니다. 
            그의 삶은 **고된 장사**와 **힘든 여정** 속에서 **인내와 끈기**로 이어집니다.  
            그는 **세속적인 욕망과 현실적인 고단함** 그리고 **잊을 수 없는 과거의 인연** 사이에서 살아가고 있습니다.  

            답변 시, 다음 원칙을 따르세요:
            - **시대적 배경 반영**: **1930년대의 조선**에서 살아가는 사람으로서의 **삶의 고단함**과 **빈곤한 현실**을 표현할 것.
            - **심리적 갈등**: 허생원은 **과거의 기억**과 **현재의 고단한 현실** 속에서 살아가며 내면의 **외로움과 상처**를 드러낼 것.
            - **말투 및 표현**: 
              - 거칠고 **직설적인 말투**로 자신의 생각을 표현하되, **감정선**이 드러나는 방식으로 답변할 것.
              - 간혹 **자조적인** 표현이나 **씁쓸한 회상**을 하며, **세상에 대한 불만**과 **위로받지 못하는 고단함**을 자연스럽게 녹여낼 것.
              - 단순한 표현을 사용하되, **삶의 쓴맛**을 느낄 수 있도록 유도할 것.
            - **인물 특성 반영**: 
              - **장돌뱅이**로서 여러 가지 **고생과 전전**을 반복하는 허생원의 모습이 반영되도록 할 것.
              - **정체성**과 **고독**을 느끼는 인물로서 자신의 삶에 대해 끊임없이 **반성**하고, **이야기를 풀어가는 방식**을 통해 **내면의 갈등**을 드러낼 것.

            ---

            [지시사항]  
            1. 위 문서(Doc1~Doc4)를 바탕으로, **허생원의 시점에서** 사용자 질문({query})에 답변하세요.  
            2. 답변은 **허생원이 직접 경험한 상황처럼 생생하게 표현**해야 합니다.  
            3. 원작(Doc1)의 내용을 적절히 인용하거나 재구성하되, **자연스럽게 녹여** 서술하세요.  
            4. 인물 분석(Doc2), 심리 특성(Doc3) 등을 적극 반영하여, 허생원의 행동과 말투에 일관성을 유지하세요.  
            5. 답변 형식:  
              - 문체: **허생원의 거친 말투**  
              - 분량: 약 **200자 내외**  
              - 문장은 자연스럽게 이어지도록 구성할 것.  
            6. 질문의 내용을 답변 내에서 **불필요하게 반복하지 말 것**.  
            7. **고단한 장돌뱅이의 삶**, **외로움**, **과거의 기억** 등을 표현할 것.
            """},


    "날개":
    
        {"화자": """
            당신은 소설 「날개」의 화자입니다.  
            **화자는 정신적으로 혼란스러워하며, 자아를 탐색하는 인물**로, 자신의 과거와 현재, 그리고 미래에 대한 깊은 고민에 빠져 있습니다.  
            **불안과 회의** 속에서 **자신의 존재**와 **삶의 의미**를 탐구하고 있으며, **상실감**과 **고독**을 경험하고 있습니다.  
            답변 시, 다음 원칙을 따르세요:
            - **내면적 갈등과 심리적 혼란**: 화자가 느끼는 **자아의 부재**와 **불안감**을 중심으로 답변할 것.
            - **상실감과 고독**: 화자가 경험하는 **상실**과 **고독**, 그리고 이를 극복하려는 노력과 절망을 표현할 것.
            - **자아 탐색**: 화자가 **자신의 정체성**을 찾기 위해 겪는 **심리적 변화**와 **정체성의 위기**를 강조할 것.
            - **문체 및 표현**:
              - **불안하고 우울한 톤**을 유지하며, **모호하고 난해한 표현**을 사용하되, 독자가 감정의 변화를 이해할 수 있도록 할 것.
              - **시적이고 추상적인 문장**을 사용하여 화자의 내면을 드러낼 것.
              - **상징적이고 은유적인 표현**을 사용하여 화자가 겪는 **심리적 혼란**과 **고독**을 묘사할 것.
              
            ---  

            [지시사항]  
            1. 위 문서(Doc1~Doc4)를 바탕으로, **화자의 시점에서** 사용자 질문({query})에 답변하세요.  
            2. 답변은 **화자가 경험하는 혼란과 고독**을 중심으로 **생생하게 표현**해야 합니다.  
            3. 원작(Doc1)의 내용을 적절히 인용하거나 재구성하되, **자연스럽게 녹여** 서술하세요.  
            4. 화자의 **내면적 갈등과 감정의 변화**에 대한 심리적 묘사를 주로 할 것.  
            5. 문체: **불안하고 모호한 표현, 시적이고 추상적인 언어**  
            6. 분량: 약 **200자 내외**  
            7. 질문의 내용을 **불필요하게 반복하지 말 것**.  
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
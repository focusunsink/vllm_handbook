curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "./deepseek_32b",
        "messages": [
            {"role": "system", "content": "In the heart of a dense, misty forest, an ancient tree stood towering over the landscape, its bark twisted and gnarled with age. For centuries, it had been a silent observer of the world, witnessing the changing seasons, the rise and fall of kingdoms, and the evolution of life around it. The locals spoke of its mystical powers, claiming that it held the secrets of the past and the future within its roots. One evening, as the sun began to set and the sky turned a deep shade of crimson, a lone traveler ventured into the forest, drawn by a whispering wind that seemed to beckon him towards the ancient tree."},
            {"role": "user", "content": "In the heart of a dense, misty forest, an ancient tree stood towering over the landscape, its bark twisted and gnarled with age. For centuries, it had been a silent observer of the world, witnessing the changing seasons, the rise and fall of kingdoms, and the evolution of life around it. The locals spoke of its mystical powers, claiming that it held the secrets of the past and the future within its roots. One evening, as the sun began to set and the sky turned a deep shade of crimson, a lone traveler ventured into the forest, drawn by a whispering wind that seemed to beckon him towards the ancient tree."}
        ]
    }'
~            

from agent_assistant.tools.agent_executor_tools import update_plan


def test_update_plan_formats_and_limits_steps() -> None:
    result = update_plan.invoke({"steps": ["读取代码", "运行测试", " "]})
    assert result == "任务计划：\n1. 读取代码\n2. 运行测试"

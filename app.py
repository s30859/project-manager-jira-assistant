import os
import streamlit as st
from jira import JIRA
from dotenv import load_dotenv
import google.generativeai as genai
import json
import time
load_dotenv()
MODEL_NAME = 'models/gemini-3-pro-preview' 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name=MODEL_NAME)

FIBONACCI = [1, 2, 3, 5, 8, 13]

def connect_jira(server, email, token):
    return JIRA(server=server, basic_auth=(email, token))

#UI Setup
st.set_page_config(page_title="Strategic Requirements Decomposition", layout="wide")
if 'workflow_result' not in st.session_state:
    st.session_state.workflow_result = None
if 'story_id_value' not in st.session_state:
    st.session_state.story_id_value = "customfield_10016"

with st.sidebar:
    st.header("🔑 Jira Authentication")
    st.caption("Enter your credentials to connect securely. Tokens are not stored.")
    jira_server = st.text_input("Jira Server URL", placeholder="https://your-domain.atlassian.net")
    jira_email = st.text_input("Jira Email")
    jira_token = st.text_input("Jira API Token", type="password")
    
    st.divider()
    st.header("Project details")
    project_key = st.text_input("Project key", placeholder="e.g., SCRUM")
    
    with st.expander("🛠️ How to find Story ID?", expanded=False):
        st.write("Click below to auto-detect the 'Story Points' field ID in your Jira instance.")
        if st.button("Detect Story Points ID", use_container_width=True):
            if jira_server and jira_email and jira_token:
                try:
                    temp_jira = connect_jira(jira_server, jira_email, jira_token)
                    fields = temp_jira.fields()
                    # Ищем поле Story Points
                    sp_fields = [f for f in fields if 'story point' in f['name'].lower()]
                    if sp_fields:
                        st.session_state.story_id_value = sp_fields[0]['id']
                        st.success(f"Found: **{sp_fields[0]['name']}** -> ID: `{sp_fields[0]['id']}`")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.warning("No Story Points field found. Check your Jira configuration.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                st.warning("Please enter your Jira credentials above first.")
                
    story_id_field = st.text_input(
        "Story ID (Field ID)", 
        value=st.session_state.story_id_value, 
        disabled=True, 
        help="This field is read-only to prevent accidental errors. Use the Auto-Detector above if you need to change it."
    )
    
    st.divider()
    st.subheader("Project specs")
    assign_to_sprint = st.checkbox("Assign to one sprint", value=False)
    sprint_id = st.text_input("Sprint ID", placeholder="e.g., 12") if assign_to_sprint else ""
    
    create_user_story = st.checkbox("Create User Story", value=True)
    assign_to_epic = st.checkbox("Assign to Epic", value=True)
    
    st.divider()
    st.write(f"**Engine:** {MODEL_NAME}")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.workflow_result = None
        st.rerun()

st.title("Strategic Requirements Decomposition")

user_requirements = st.text_area(
    "Requirements Input", 
    placeholder="Describe technical scope and business goals...", 
    height=200, 
    label_visibility="collapsed"
)

# EPIC LOGIC
selected_epic_key = None
new_epic_title = ""

if assign_to_epic:
    st.write("### Epic Management")
    if project_key and jira_server and jira_email and jira_token:
        try:
            jira = connect_jira(jira_server, jira_email, jira_token)
            existing_epics = jira.search_issues(f'project = "{project_key}" AND issuetype = Epic ORDER BY created DESC', maxResults=15)
            if existing_epics:
                epic_options = {f"{e.key}: {e.fields.summary}": e.key for e in existing_epics}
                epic_options["➕ Create a new Epic..."] = "CREATE_NEW"
                choice = st.selectbox("Select Epic", options=list(epic_options.keys()), label_visibility="collapsed")
                selected_epic_key = epic_options[choice]
            else:
                selected_epic_key = "CREATE_NEW"

            if selected_epic_key == "CREATE_NEW":
                new_epic_title = st.text_input("New Epic Title", placeholder="Enter title for new Epic...", label_visibility="collapsed")
        except:
            selected_epic_key = "CREATE_NEW"
            new_epic_title = st.text_input("New Epic Title", placeholder="Enter title for new Epic...", label_visibility="collapsed")
    else:
        st.info("Enter Jira Credentials and Project Key in the sidebar to load Epics.")

#AGENTIC WORKFLOW WITH FEEDBACK LOOP 
st.write("")
if st.button("🚀 Run AI Agentic Pipeline", use_container_width=True):
    if user_requirements and project_key:
        try:
            with st.status("Initializing Multi-Agent Pipeline...", expanded=True) as status:
                max_iterations = 3
                iteration = 0
                workflow_passed = False
                
                current_pm_output = user_requirements
                tech_tasks_json = ""
                tech_tasks = []
                audit_result = {}
                culprit = "NONE"
                feedback_report = ""

                while iteration < max_iterations and not workflow_passed:
                    iteration += 1
                    st.write(f"### 🔄 Iteration {iteration}")

                    # STAGE 1: PM
                    if create_user_story and (iteration == 1 or culprit == "PM"):
                        st.write("📝 Stage 1: PM agent is drafting User Stories...")
                        pm_prompt = "Act as a PM. Convert input into User Stories. Format: 'As a [user], I want [action] so that [value]'. Return JSON array: [{'summary': '...', 'ac': '...'}]"
                        if culprit == "PM":
                            st.write(f"   *Applying QA feedback to PM...*")
                            pm_prompt += f"\n\nCRITICAL FIX REQUIRED: The QA Auditor rejected your previous output. Feedback: {feedback_report}. Please fix these issues."
                        pm_resp = model.generate_content(pm_prompt + "\n\nInput: " + user_requirements)
                        current_pm_output = pm_resp.text.strip().replace('```json', '').replace('```', '')

                    # STAGE 2: ARCHITECT
                    if iteration == 1 or culprit in ["PM", "ARCHITECT"]:
                        st.write("⚙️ Stage 2: Architect agent is building technical plan...")
                        arch_prompt = (
                            "Act as a Lead Architect. Decompose input into technical tasks. "
                            "Preserve the exact original text of the 'summary' from the input. "
                            "Return JSON array: [{'summary': '...', 'description': '...', 'sp': 3, 'subtasks': ['...']}]"
                        )
                        if culprit == "ARCHITECT":
                            st.write(f"   *Applying QA feedback to Architect...*")
                            arch_prompt += f"\n\nCRITICAL FIX REQUIRED: The QA Auditor rejected your tech plan. Feedback: {feedback_report}. Please improve the technical subtasks or description based on this."
                        arch_resp = model.generate_content(arch_prompt + "\n\nInput: " + current_pm_output)
                        tech_tasks_json = arch_resp.text.strip().replace('```json', '').replace('```', '')
                        tech_tasks = json.loads(tech_tasks_json)

                    # STAGE 3: AUDITOR
                    st.write("⚖️ Stage 3: QA Auditor is evaluating the pipeline state...")
                    audit_prompt = (
                        "Act as QA Auditor. Compare original requirements with the tech plan. "
                        "If business logic/requirements are missing, culprit is 'PM'. "
                        "If technical details are weak or not actionable, culprit is 'ARCHITECT'. "
                        "If everything is perfect, status is 'PASS' and culprit is 'NONE'. "
                        "Return JSON: {'status': 'PASS/FAIL', 'culprit': 'PM/ARCHITECT/NONE', 'report': '...'}"
                    )
                    audit_resp = model.generate_content(f"{audit_prompt}\n\nReq: {user_requirements}\nPM Stories: {current_pm_output}\nTech Plan: {tech_tasks_json}")
                    audit_result = json.loads(audit_resp.text.strip().replace('```json', '').replace('```', ''))

                    if audit_result['status'] == 'PASS':
                        workflow_passed = True
                        st.write("✅ **QA Auditor passed the architecture!**")
                    else:
                        culprit = audit_result.get('culprit', 'ARCHITECT')
                        feedback_report = audit_result.get('report', 'Unknown error.')
                        st.error(f"⚠️ **QA Rejected.** Culprit: {culprit}. Triggering self-correction loop...")
                        time.sleep(1)

                if workflow_passed:
                    status.update(label="Pipeline finished successfully!", state="complete")
                else:
                    status.update(label=f"Pipeline stopped after {max_iterations} iterations. Check Auditor report.", state="error")
                
                st.session_state.workflow_result = {"tasks": tech_tasks, "audit": audit_result}
                
        except Exception as e:
            st.error(f"Workflow Execution Error: {e}")
    else:
        st.error("Please provide Requirements and Project Key.")

# REVIEW & DEPLOY
if st.session_state.workflow_result:
    audit = st.session_state.workflow_result['audit']
    
    expander_title = f"⚖️ Final Auditor Verdict: {audit['status']}"
    if audit['status'] != 'PASS':
         expander_title += f" (Culprit: {audit.get('culprit', 'N/A')})"
         
    with st.expander(expander_title, expanded=True):
        st.write(audit['report'])

    backlog = st.session_state.workflow_result['tasks']
    indices_to_remove = []

    for idx, story in enumerate(backlog):
        d_idx = idx + 1 
        with st.container(border=True):
            h_col, del_col = st.columns([12, 1])
            with h_col: st.markdown(f"### Item #{d_idx}")
            with del_col:
                if st.button("❌", key=f"del_{idx}"): indices_to_remove.append(idx)

            c1, c2 = st.columns([2, 1])
            with c1:
                story['summary'] = st.text_input(f"Summary #{d_idx}", value=story['summary'], key=f"sum_{idx}")
                story['description'] = st.text_area(f"Technical Plan #{d_idx}", value=story['description'], height=150, key=f"desc_{idx}")
                for tidx, sub in enumerate(story['subtasks']):
                    story['subtasks'][tidx] = st.text_input(f"Sub-task {d_idx}.{tidx+1}", value=sub, key=f"sub_{idx}_{tidx}")
            with c2:
                st.write("**Points**")
                story['sp'] = st.segmented_control(
                    label=f"sp_{idx}", 
                    options=FIBONACCI, 
                    default=story['sp'] if story['sp'] in FIBONACCI else 3, 
                    key=f"choice_{idx}", 
                    label_visibility="collapsed"
                ) or 3

    if indices_to_remove:
        for i in sorted(indices_to_remove, reverse=True): backlog.pop(i)
        st.rerun()

    if st.button("🚀 Push Verified Plan to Jira", type="primary", use_container_width=True):
        if not (jira_server and jira_email and jira_token):
            st.error("Please enter Jira Credentials in the sidebar to deploy.")
        else:
            try:
                jira = connect_jira(jira_server, jira_email, jira_token)
                with st.status("Deploying to Jira..."):
                    
                    p_info = jira.project(project_key)
                    i_types = {it.name.lower(): it.id for it in p_info.issueTypes}
                    
                    epic_type_id = i_types.get('epic')
                    story_type_id = i_types.get('story') or i_types.get('task')
                    subtask_type_id = i_types.get('sub-task') or i_types.get('subtask')
                    
                    if not story_type_id:
                        raise Exception(f"Issue type 'Story' or 'Task' not found in project {project_key}")
                    if not subtask_type_id:
                        raise Exception(f"Issue type 'Sub-task' not found in project {project_key}")

                    parent_key = None
                    if assign_to_epic:
                        if selected_epic_key == "CREATE_NEW":
                            if not epic_type_id:
                                raise Exception(f"Epic issue type not enabled in project {project_key}")
                                
                            e_dict = {'project': project_key, 'summary': new_epic_title, 'issuetype': {'id': epic_type_id}}
                            e_name_f = next((f['id'] for f in jira.fields() if 'Epic Name' in f['name']), None)
                            if e_name_f: e_dict[e_name_f] = new_epic_title[:30]
                            parent_key = jira.create_issue(fields=e_dict).key
                        else:
                            parent_key = selected_epic_key

                    for s in backlog:
                        if not s['summary'] or not s['summary'].strip(): continue

                        s_fields = {
                            'project': project_key, 'summary': f"[AI] {s['summary'][:250]}",
                            'description': s['description'], 'issuetype': {'id': story_type_id},
                            st.session_state.story_id_value: float(s['sp'])
                        }
                        if parent_key: s_fields['parent'] = {'key': parent_key}
                        if assign_to_sprint and sprint_id: s_fields['customfield_10020'] = int(sprint_id)
                        
                        st_issue = jira.create_issue(fields=s_fields)
                        
                        for sub_t in s.get('subtasks', []):
                            if sub_t and sub_t.strip():
                                jira.create_issue({
                                    'project': project_key, 'summary': sub_t[:250], 
                                    'issuetype': {'id': subtask_type_id}, 'parent': {'id': st_issue.id}
                                })
                st.success("Successfully pushed to Jira!")
                st.session_state.workflow_result = None
            except Exception as e:
                st.error(f"Jira Error: {e}")

st.divider()
st.caption("Agentic AI Sprint Architect 2026")